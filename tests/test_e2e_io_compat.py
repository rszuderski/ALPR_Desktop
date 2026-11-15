"""Exact approved SHA pair, one experiment, no writes to real artifacts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from auto_annotation_tool.ranking import e2e_io_compat as compat
from auto_annotation_tool.ranking import e2e_eval_runner as runner
from auto_annotation_tool.ranking import e2e_artifacts as store
from auto_annotation_tool.ranking import e2e_eval_report as reports
from e2e_fixture import build_env


@pytest.fixture
def approved(monkeypatch):
    root = Path(compat.__file__).resolve().parents[2]
    run = (root / compat.RUN_RELATIVE).resolve()
    bundle = root / compat.BUNDLE_RELATIVE
    with zipfile.ZipFile(bundle / 'original_run_binding.zip') as archive:
        # Immutable archival bytes, never use live progress/pending in tests.
        files = {run / name: archive.read(name) for name in archive.namelist()}
    original_read, original_exists = Path.read_bytes, Path.exists
    def read(path):
        key = path.resolve()
        if key in files:
            return files[key]
        if key.is_relative_to(run):
            raise FileNotFoundError('No real experiment access from test: '+str(key))
        return original_read(path)
    def exists(path):
        key = path.resolve()
        return key in files if key.is_relative_to(run) else original_exists(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(Path, 'exists', exists)
    protocol = json.loads(files[run / 'protocol.json'])
    return SimpleNamespace(root=root, run=run, bundle=bundle, files=files, protocol=protocol,
                           actual=runner._code_receipt(), document_sha=runner.IO_COMPAT_DOCUMENT_SHA)


def check(env, **kwargs):
    return compat.verify_runtime_code(kwargs.get('protocol', env.protocol), kwargs.get('actual', env.actual),
                                      kwargs.get('run', env.run), kwargs.get('document_sha', env.document_sha))


def test_only_exact_patch_and_exact_wiring_are_accepted_read_only(approved):
    before = deepcopy(approved.files)
    certificate = check(approved)
    assert certificate['historical_code_sha256'] == approved.protocol['code_sha256']
    assert certificate['effective_code_sha256'] == approved.actual
    assert certificate['document']['writer_sha256'] == {'original': compat.OLD_MT_SHA, 'corrected': compat.NEW_MT_SHA}
    assert approved.files == before
    assert json.loads(approved.files[approved.run/'approval.json'])['operator_signature'] == 'rs'
    assert (approved.run/compat.APPLIED_FILE) not in approved.files


@pytest.mark.parametrize('kind', ['new_start', 'other_name', 'same_name_other_folder'])
def test_exception_never_applies_to_a_new_or_other_experiment(approved, kind):
    path = {'new_start': None, 'other_name': approved.run.with_name('EVAL396_E2E_run_OTHER'),
            'same_name_other_folder': approved.run.parent/'other'/approved.run.name}[kind]
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        check(approved, run=path)


@pytest.mark.parametrize('relative', [
    compat.MT_FILE, compat.E2E_FILE, compat.HELPER_FILE, compat.REPORT_FILE, compat.DOCUMENT_RELATIVE,
    'auto_annotation_tool/ranking/e2e_eval_metrics.py', 'auto_annotation_tool/ranking/mt_eval_metrics.py',
    'auto_annotation_tool/ranking/e2e_artifacts.py', 'auto_annotation_tool/rectification/plate_rectifier.py',
    'auto_annotation_tool/rectification/quad_geometry.py', 'auto_annotation_tool/image_orientation.py',
    'auto_annotation_tool/registration_text.py', 'auto_annotation_tool/character_recognition/reading_order.py',
])
def test_any_other_code_sha_is_rejected(approved, relative):
    actual = deepcopy(approved.actual)
    actual[str((approved.root/relative).resolve())] = '0'*64
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        check(approved, actual=actual)


@pytest.mark.parametrize('kind', ['old_writer', 'unknown_old_sha', 'missing_file', 'extra_file', 'protocol'])
def test_no_global_ignore_or_receipt_rewrite(approved, kind):
    actual, protocol = deepcopy(approved.actual), deepcopy(approved.protocol)
    mt = str((approved.root/compat.MT_FILE).resolve())
    if kind == 'old_writer': actual[mt] = compat.OLD_MT_SHA
    elif kind == 'unknown_old_sha': protocol['code_sha256'][mt] = 'f'*64
    elif kind == 'missing_file': actual.pop(mt)
    elif kind == 'extra_file': actual['unexpected.py'] = 'f'*64
    else: protocol['inference']['conf'] = .99
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        check(approved, actual=actual, protocol=protocol)


@pytest.mark.parametrize('name', ['progress.json', 'pending.json', 'approval.json', 'inputs.json', 'provenance.json'])
def test_changed_binding_or_recovery_baseline_is_rejected(approved, name):
    approved.files[approved.run/name] += b' '
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        check(approved)


def test_changed_pending_raw_is_rejected(approved):
    pending = json.loads(approved.files[approved.run/'pending.json'])
    approved.files[approved.run/pending['file']] += b' '
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        check(approved)


@pytest.mark.parametrize('name', ['compatibility.json', 'original_mt_eval_runner.py.txt',
                                 'corrected_mt_eval_runner.py.txt', 'original_e2e_eval_runner.py.txt'])
def test_changed_evidence_is_rejected(approved, monkeypatch, name):
    original = Path.read_bytes
    target = (approved.bundle/name).resolve()
    def read(path):
        raw = original(path)
        return raw+b'\n# unauthorized change' if path.resolve() == target else raw
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        check(approved)


def test_writer_scope_and_runner_integration_reject_other_bytes(approved):
    original = (approved.bundle/'original_mt_eval_runner.py.txt').read_bytes()
    fixed = (approved.bundle/'corrected_mt_eval_runner.py.txt').read_bytes()
    compat.verify_writer_scope(original, fixed)
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        compat.verify_writer_scope(original, fixed+b'\n# another change')
    before = (approved.bundle/'original_e2e_eval_runner.py.txt').read_bytes()
    expected = compat.patched_runner_bytes(before, approved.document_sha)
    assert hashlib.sha256(expected).hexdigest() == approved.actual[str((approved.root/compat.E2E_FILE).resolve())]
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        compat.patched_runner_bytes(before+b'\n', approved.document_sha)


def test_exact_code_equality_stays_strict_default_without_exception(approved):
    assert check(approved, actual=approved.protocol['code_sha256'], run=None) is None


def test_applied_provenance_allows_later_valid_progress_but_cannot_be_rewritten(approved):
    certificate = check(approved)
    applied = approved.run/compat.APPLIED_FILE
    approved.files[applied] = store.canonical(certificate)+b'\n'
    # The original prefix is mandatory; native validation checks later rows too.
    ledger = json.loads(approved.files[approved.run/'progress.json'])
    pending = json.loads(approved.files[approved.run/'pending.json'])
    ledger['records'][pending['file']] = pending['sha256']
    approved.files.pop(approved.run/'pending.json')
    approved.files[approved.run/'progress.json'] = store.canonical(ledger)+b'\n'
    assert check(approved) == certificate
    assert compat.read_application(approved.run, approved.protocol, approved.document_sha) == certificate
    changed = deepcopy(certificate); changed['effective_code_sha256']['foreign.py'] = '0'*64
    approved.files[applied] = store.canonical(changed)+b'\n'
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        check(approved)
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        compat.read_application(approved.run, approved.protocol, approved.document_sha)


@pytest.mark.parametrize('change', ['drop', 'sha'])
def test_applied_compatibility_never_allows_losing_prior_results(approved, change):
    certificate = check(approved)
    approved.files[approved.run/compat.APPLIED_FILE] = store.canonical(certificate)+b'\n'
    ledger = json.loads(approved.files[approved.run/'progress.json'])
    name = next(iter(ledger['records']))
    if change == 'drop': ledger['records'].pop(name)
    else: ledger['records'][name] = '0'*64
    approved.files[approved.run/'progress.json'] = store.canonical(ledger)+b'\n'
    with pytest.raises(ValueError, match='E2E_RUNTIME_CODE_CHANGED'):
        check(approved)


def test_application_addendum_is_idempotent_and_does_not_repair_journal(approved, tmp_path, monkeypatch):
    certificate = check(approved)
    certificate['run_dir'] = str(tmp_path.resolve())  # Synthetic application target only.
    pending = tmp_path/'pending.json';pending.write_bytes(b'TEST_PENDING_UNCHANGED')
    monkeypatch.setattr(compat, '_check_initial_recovery_state', lambda *_: None)
    compat.record_application(tmp_path, certificate)
    file = tmp_path/compat.APPLIED_FILE; before = file.read_bytes(), file.stat().st_mtime_ns
    compat.record_application(tmp_path, certificate)
    assert (file.read_bytes(), file.stat().st_mtime_ns) == before
    assert pending.read_bytes() == b'TEST_PENDING_UNCHANGED'
    bad = deepcopy(certificate);bad['document_sha256'] = '0'*64
    with pytest.raises(ValueError, match='IMMUTABLE_FILE_CONFLICT'):
        compat.record_application(tmp_path, bad)


def test_native_resume_records_provenance_before_recover_and_skips_saved_forwards(tmp_path, monkeypatch):
    env = build_env(tmp_path, monkeypatch)
    partial = runner.run_campaign(env.prepared, env.approval, cancelled=lambda:len(env.calls)==1)
    path = Path(partial['run_dir']);before={p:p.read_bytes() for p in (path/'raw').rglob('*.json')}
    original_verify, original_recover = compat.verify_runtime_code, store.recover
    certificate = {'run_dir':str(path),'TEST_ONLY':True};events=[]
    def verify(protocol, actual, resume_dir, document_sha):
        original_verify(protocol, actual, resume_dir, document_sha)
        assert Path(resume_dir)==path
        return certificate
    def recover(*a,**kw):
        assert (path/compat.APPLIED_FILE).exists()
        events.append('recover');return original_recover(*a,**kw)
    monkeypatch.setattr(compat,'verify_runtime_code',verify)
    monkeypatch.setattr(compat,'_check_initial_recovery_state',lambda *_:None)
    monkeypatch.setattr(compat,'read_application',lambda p,*_:store.read_json(Path(p)/compat.APPLIED_FILE) if (Path(p)/compat.APPLIED_FILE).exists() else None)
    monkeypatch.setattr(store,'recover',recover)
    completed=runner.run_campaign(env.prepared,env.approval,resume_dir=path)
    assert completed['status']=='COMPLETE' and len(env.calls)==3 and events==['recover']
    assert all(p.read_bytes()==v for p,v in before.items())
    assert completed['io_compatibility']==certificate
    assert compat.APPLIED_FILE in store.read_json(path/'completion_manifest.json')['files']
    runner.run_campaign(env.prepared,env.approval,resume_dir=path)
    assert len(env.calls)==3 and events==['recover']


def test_report_includes_separate_provenance_without_rewriting_protocol(approved, tmp_path, monkeypatch):
    env = build_env(tmp_path, monkeypatch)
    report = runner.run_campaign(env.prepared,env.approval)
    report['io_compatibility']=check(approved)
    before=deepcopy(report['protocol'])
    monkeypatch.setattr(reports,'load_campaign',lambda *_:report)
    output=reports.export_report(report['run_dir'],tmp_path/'exports')
    assert report['protocol']==before
    assert store.read_json(output/compat.APPLIED_FILE)==report['io_compatibility']
    assert store.read_json(output/'provenance.json')['io_compatibility']==report['io_compatibility']
    assert compat.OLD_MT_SHA in (output/'report.md').read_text(encoding='utf8')
    assert compat.NEW_MT_SHA in (output/'report.md').read_text(encoding='utf8')
