"""Crash-boundary tests using synthetic bytes and a fake pixel-only backend."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from auto_annotation_tool.ranking import mt_eval_runner as runner
from auto_annotation_tool.ranking import mt_geometry_review as gate
from auto_annotation_tool.ranking.eval396 import EvidenceReader, digest
from test_mt_eval_runner import runner_env
from test_mt_geometry_review import review


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fail_once(monkeypatch, predicate, message):
    original = runner._write_json
    fired = []
    def write(path, payload, **kwargs):
        if not fired and predicate(Path(path), payload):
            fired.append(True)
            raise OSError(message)
        return original(path, payload, **kwargs)
    monkeypatch.setattr(runner, '_write_json', write)
    return fired


@pytest.mark.parametrize('boundary', ['raw_publication', 'ledger_publication'])
def test_pending_journal_recovers_completed_forward_without_repeating_it(runner_env, monkeypatch, boundary):
    if boundary == 'raw_publication':
        condition = lambda path, payload: path.parent.name == 'raw'
    else:
        condition = lambda path, payload: path.name == 'progress.json' and bool(payload.get('records'))
    fired = fail_once(monkeypatch, condition, 'SIMULATED_CRASH_'+boundary)
    partial = runner.run_campaign(runner_env.prepared)
    path = Path(partial['run_dir'])
    assert fired and partial['status'] == 'PARTIAL' and partial['recovery_required']
    assert len(runner_env.backend_calls) == 1
    assert partial['models'][0]['rows'] == []
    assert len(partial['models'][0]['recoverable_rows']) == 1
    pending = path/'MT-n/pending.json'
    assert pending.exists()
    protected = {p: p.read_bytes() for p in path.glob('*/raw/*.json')}
    result = runner.run_campaign(runner_env.prepared, resume_dir=path)
    assert result['status'] == 'COMPLETE' and not result['recovery_required']
    assert len(runner_env.backend_calls) == 9
    assert runner_env.backend_calls.count(('MT-n', 0)) == 1
    assert not pending.exists()
    assert all(p.read_bytes() == original for p, original in protected.items())


def test_pending_cleanup_failure_after_last_row_does_not_strand_complete_model(runner_env, monkeypatch):
    original = Path.unlink
    fired = []
    def unlink(path, *args, **kwargs):
        if path.name == 'pending.json' and path.parent.name == 'MT-n' and len(runner_env.backend_calls) == 3 and not fired:
            fired.append(True)
            raise PermissionError('SIMULATED_PENDING_CLEANUP_FAILURE')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    partial = runner.run_campaign(runner_env.prepared)
    assert fired and partial['status'] == 'PARTIAL'
    assert partial['models'][0]['status'] == 'PARTIAL'
    assert partial['models'][0]['metrics']['status'] == 'COMPLETE'
    assert partial['models'][0]['recovery_required']
    assert partial['models'][0]['recoverable_rows'] == []
    result = runner.run_campaign(runner_env.prepared, resume_dir=partial['run_dir'])
    assert result['status'] == 'COMPLETE' and len(runner_env.backend_calls) == 9
    assert [e for e in runner_env.event_log if e == ('load', 'MT-n')] == [('load', 'MT-n')]


def test_completion_manifest_publication_then_status_failure_is_idempotent(runner_env, monkeypatch):
    fired = fail_once(monkeypatch, lambda p, v: p.name == 'status.json' and v.get('status') == 'COMPLETE',
                      'SIMULATED_FINAL_STATUS_CRASH')
    partial = runner.run_campaign(runner_env.prepared)
    assert fired and partial['status'] == 'PARTIAL'
    assert all(m['status'] == 'COMPLETE' for m in partial['models'])
    path = Path(partial['run_dir'])
    seal = path/'completion_manifest.json'
    before = seal.read_bytes(), seal.stat().st_mtime_ns
    model_files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.glob('MT-*/*.json')}
    result = runner.run_campaign(runner_env.prepared, resume_dir=path)
    assert result['status'] == 'COMPLETE' and len(runner_env.backend_calls) == 9
    assert (seal.read_bytes(), seal.stat().st_mtime_ns) == before
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in model_files.items())


def test_resuming_fresh_preflight_ignores_only_nonfingerprinted_created_at(runner_env):
    partial = runner.run_campaign(runner_env.prepared, cancelled=lambda: len(runner_env.backend_calls) >= 1)
    prepared = deepcopy(runner_env.prepared)
    prepared['preflight']['created_at'] = 'a fresh preflight timestamp'
    result = runner.run_campaign(prepared, resume_dir=partial['run_dir'])
    assert result['status'] == 'COMPLETE' and len(runner_env.backend_calls) == 9


def test_unjournaled_orphan_is_not_silently_adopted(runner_env):
    partial = runner.run_campaign(runner_env.prepared, cancelled=lambda: len(runner_env.backend_calls) >= 1)
    path = Path(partial['run_dir'])
    ledger_path = path/'MT-n/progress.json'
    ledger = _json(ledger_path)
    ledger.update(records={}, summary_sha256=None)
    runner._write_json(ledger_path, ledger)
    with pytest.raises(ValueError, match='MT_UNJOURNALED_ORPHAN_RAW_RECORD'):
        runner.run_campaign(runner_env.prepared, resume_dir=path)
    assert len(runner_env.backend_calls) == 1


def test_missing_indexed_raw_cannot_be_repredicted(runner_env):
    partial = runner.run_campaign(runner_env.prepared, cancelled=lambda: len(runner_env.backend_calls) >= 1)
    path = Path(partial['run_dir'])
    next((path/'MT-n/raw').glob('*.json')).unlink()
    with pytest.raises(ValueError, match='MT_MISSING_INDEXED_RAW_RECORD'):
        runner.run_campaign(runner_env.prepared, resume_dir=path)
    assert len(runner_env.backend_calls) == 1


def test_rehashed_pending_effective_parameter_change_is_rejected(runner_env, monkeypatch):
    fail_once(monkeypatch, lambda p, v: p.parent.name == 'raw', 'SIMULATED_RAW_CRASH')
    partial = runner.run_campaign(runner_env.prepared)
    path = Path(partial['run_dir'])/'MT-n/pending.json'
    pending = _json(path)
    pending['row']['effective_inference']['augment'] = True
    pending['row']['record_sha256'] = digest(runner._without(pending['row'], 'record_sha256'))
    pending['raw_file_sha256'] = __import__('hashlib').sha256(runner.canonical(pending['row'])+b'\n').hexdigest()
    pending['transaction_sha256'] = digest(runner._without(pending, 'transaction_sha256'))
    runner._write_json(path, pending)
    with pytest.raises(ValueError, match='MT_EFFECTIVE_PARAMETER_CHANGED'):
        runner.run_campaign(runner_env.prepared, resume_dir=partial['run_dir'])
    assert len(runner_env.backend_calls) == 1


@pytest.mark.parametrize('change', ['remove', 'add'])
def test_completion_seal_requires_exact_contract_file_coverage(runner_env, change):
    result = runner.run_campaign(runner_env.prepared)
    path = Path(result['run_dir'])/'completion_manifest.json'
    seal = _json(path)
    if change == 'remove':
        seal['files'].pop('MT-n/summary.json')
    else:
        seal['files']['status.json'] = runner._sha(path.parent/'status.json')
    runner._write_json(path, seal)
    with pytest.raises(ValueError, match='MT_COMPLETION_FILE_COVERAGE_CHANGED'):
        runner.load_campaign(path.parent)


def test_exclusive_json_publication_never_overwrites_existing_target(tmp_path):
    path = tmp_path/'record.json'
    runner._write_json(path, {'first': True}, exclusive=True)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner._write_json(path, {'replacement': True}, exclusive=True)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('attack', ['approval', 'valid_gt_rehash', 'reference_digest'])
def test_real_snapshot_binding_rejects_resealed_false_approval_or_gt(review, tmp_path, attack):
    _, decision = gate.save_operator_decision(review, tmp_path.parent/(tmp_path.name+'_contract'),
        operator='TEST FIXTURE ONLY', reviewed_scenes=review['review_scene_hashes'], accepted=True)
    report = deepcopy(review)
    protocol = {'models': deepcopy(report['models']), 'metric_counts': report['metric_counts'],
                'metric_policy_sha256': report['metric_policy_sha256'], 'metric_policy_id': report['metric_policy']['id'],
                'source_receipt_sha256': digest(report['source_files'])}
    runner._verify_campaign_geometry(protocol, report, decision, report['source_files'], EvidenceReader())
    if attack == 'approval':
        decision['decision'] = 'REJECTED'
        decision['contract_sha256'] = digest(runner._without(decision, 'contract_sha256'))
        reason = 'MT_STORED_GEOMETRY_APPROVAL_REQUIRED'
    else:
        report['scenes'][0]['plates'][0]['quad'][0][0] += .01
        reason = 'MT_REFERENCE_FINGERPRINT_CHANGED'
        if attack == 'valid_gt_rehash':
            report.update(gate.assess_metric_scope(report['scenes'], report['metric_policy']))
            report['review_sha256'] = gate.review_fingerprint(report)
            decision['review_sha256'] = report['review_sha256']
            decision['contract_sha256'] = digest(runner._without(decision, 'contract_sha256'))
            reason = 'REPORT_DIFFERS_FROM_FROZEN_GT'
    with pytest.raises(ValueError, match=reason):
        runner._verify_campaign_geometry(protocol, report, decision, report['source_files'], EvidenceReader())
