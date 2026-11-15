"""Bounded Windows publication retry; all writes use isolated test directories."""
import errno
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from auto_annotation_tool.ranking import mt_eval_runner as mt
from auto_annotation_tool.ranking import e2e_eval_runner as e2e
from auto_annotation_tool.ranking import e2e_artifacts as store
from e2e_fixture import build_env
from test_z4_e2e_ui import ui, spin, prepare_and_sign


def win_error(code):
    error = PermissionError(errno.EACCES, 'TEST_WINDOWS_PUBLICATION_DENIED')
    error.winerror = code
    return error


def atomic_os(monkeypatch, replace, *, name='nt', fsync=None):
    monkeypatch.setattr(mt, 'os', SimpleNamespace(name=name, replace=replace,
        fsync=fsync or os.fsync, rename=os.rename, link=os.link))


@pytest.mark.parametrize('code', [5, 32, 33])
def test_transient_windows_error_retries_same_fsynced_bytes(tmp_path, monkeypatch, code):
    target = tmp_path / 'progress.json'
    target.write_bytes(b'{"old":true}\n')
    expected = mt.canonical({'records': {'raw.json': 'test-sha'}}) + b'\n'
    events, attempts, delays = [], [], []
    def fsync(fd):
        os.fsync(fd); events.append('fsync')
    def replace(source, destination):
        assert events == ['fsync']
        assert Path(source).read_bytes() == expected
        assert target.read_bytes() == b'{"old":true}\n'
        attempts.append(str(source))
        if len(attempts) < 3:
            raise win_error(code)
        os.replace(source, destination)
    atomic_os(monkeypatch, replace, fsync=fsync)
    monkeypatch.setattr(mt.time, 'sleep', delays.append)
    assert mt._write_json(target, {'records': {'raw.json': 'test-sha'}}) == hashlib.sha256(expected).hexdigest()
    assert len(set(attempts)) == 1 and len(attempts) == 3
    assert delays == [.025, .05] and target.read_bytes() == expected
    assert not list(tmp_path.glob('*.tmp.*'))


@pytest.mark.parametrize('code', [5, 32, 33])
def test_permanent_windows_error_is_bounded_and_preserves_old_ledger(tmp_path, monkeypatch, code):
    target = tmp_path / 'progress.json'; target.write_bytes(b'OLD_LEDGER')
    error = win_error(code); attempts, delays = [], []
    def replace(*args):
        attempts.append(args); raise error
    atomic_os(monkeypatch, replace)
    monkeypatch.setattr(mt.time, 'sleep', delays.append)
    with pytest.raises(PermissionError) as caught:
        mt._write_json(target, {'new': True})
    assert caught.value is error and len(attempts) == 6
    assert delays == [.025, .05, .1, .2, .4] and sum(delays) < 1
    assert target.read_bytes() == b'OLD_LEDGER' and not list(tmp_path.glob('*.tmp.*'))


@pytest.mark.parametrize('platform,code', [('nt', None), ('nt', 2), ('nt', 17), ('nt', 112), ('posix', 5)])
def test_other_errors_do_not_retry(tmp_path, monkeypatch, platform, code):
    error = win_error(code); attempts = []
    def replace(*args):
        attempts.append(args); raise error
    atomic_os(monkeypatch, replace, name=platform)
    monkeypatch.setattr(mt.time, 'sleep', lambda *_: pytest.fail('Unexpected retry'))
    with pytest.raises(OSError) as caught:
        mt._write_json(tmp_path / 'progress.json', {})
    assert caught.value is error and len(attempts) == 1


def test_exclusive_conflict_is_not_retried_or_overwritten(tmp_path, monkeypatch):
    target = tmp_path / 'raw.json'; target.write_bytes(b'ORIGINAL_RAW')
    monkeypatch.setattr(mt.time, 'sleep', lambda *_: pytest.fail('Exclusive conflict must fail immediately'))
    with pytest.raises(FileExistsError):
        mt._write_json(target, {'new': True}, exclusive=True)
    assert target.read_bytes() == b'ORIGINAL_RAW'


@pytest.mark.skipif(os.name != 'nt', reason='Real Windows sharing semantics')
def test_real_windows_reader_blocks_replace_until_handle_closes(tmp_path, monkeypatch, record_testsuite_property):
    target = tmp_path / 'progress.json'; target.write_bytes(b'OLD')
    reader = target.open('rb'); actual_replace = os.replace; codes = []
    def replace(*args):
        try:
            return actual_replace(*args)
        except OSError as exc:
            codes.append(exc.winerror); raise
    def release(_delay):
        assert reader.read() == b'OLD'
        reader.close()
    atomic_os(monkeypatch, replace)
    monkeypatch.setattr(mt.time, 'sleep', release)
    try:
        mt._write_json(target, {'new': True})
    finally:
        reader.close()
    assert codes and set(codes) <= {5, 32, 33}
    record_testsuite_property('native_reader_replace_winerror', str(codes))
    assert json.loads(target.read_text()) == {'new': True}


@pytest.mark.parametrize('transient', [True, False])
def test_e2e_raw_before_ledger_failure_recovers_without_repeating_forward(tmp_path, monkeypatch, transient):
    env = build_env(tmp_path, monkeypatch)
    actual_replace = os.replace; attempts = []
    def replace(source, target):
        target = Path(target)
        pending = target.parent / 'pending.json'
        if target.name == 'progress.json' and pending.exists():
            transaction = store.read_json(pending)
            if transaction['record']['schema'] == 'alpr.e2e.prediction.v1' and len(env.calls) == 1:
                assert (target.parent / transaction['file']).exists()
                attempts.append(str(source))
                if not transient or len(attempts) == 1:
                    raise win_error(5)
        return actual_replace(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(mt.os, 'replace', replace)
        patch.setattr(mt.time, 'sleep', lambda _: None)
        if transient:
            result = e2e.run_campaign(env.prepared, env.approval)
            assert result['status'] == 'COMPLETE' and len(env.calls) == 3
            assert len(attempts) == 2
            return
        with pytest.raises(RuntimeError, match='E2E_PARTIAL'):
            e2e.run_campaign(env.prepared, env.approval)
    run_dir = next(Path(env.context['output_root']).glob('EVAL396_E2E_run_*'))
    inspected = e2e.load_campaign(run_dir)
    assert len(attempts) == 6 and len(env.calls) == 1
    assert inspected['status'] == 'PARTIAL' and inspected['recovery_required']
    assert inspected['ledger']['pending']['file'] not in inspected['ledger']['ledger']['records']
    saved = {p: p.read_bytes() for p in (run_dir / 'raw').rglob('*.json')}
    result = e2e.run_campaign(env.prepared, env.approval, resume_dir=run_dir)
    assert result['status'] == 'COMPLETE' and len(env.calls) == 3
    assert all(p.read_bytes() == data for p, data in saved.items())
    assert not (run_dir / 'pending.json').exists()
    assert e2e.run_campaign(env.prepared, env.approval, resume_dir=run_dir)['status'] == 'COMPLETE'
    assert len(env.calls) == 3


def test_recovery_rejects_raw_sha_conflict_without_touching_ledger(tmp_path, monkeypatch):
    env = build_env(tmp_path, monkeypatch)
    original_write = store._write_json
    def fail_ledger(path, payload, **kw):
        if Path(path).name == 'progress.json':
            raise win_error(5)
        return original_write(path, payload, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(store, '_write_json', fail_ledger)
        with pytest.raises(RuntimeError, match='E2E_PARTIAL'):
            e2e.run_campaign(env.prepared, env.approval)
    run_dir = next(Path(env.context['output_root']).glob('EVAL396_E2E_run_*'))
    allowed = store.expected_paths(env.prepared['inputs'], env.prepared['protocol']['pairs'])
    inspected = store.inspect_ledger(run_dir, allowed)
    ledger_before = (run_dir / 'progress.json').read_bytes()
    (run_dir / inspected['pending']['file']).write_bytes(b'CORRUPTED_TEST_COPY')
    with pytest.raises(ValueError, match='E2E_RECOVERY_RAW_CHANGED'):
        store.recover(run_dir, inspected)
    assert (run_dir / 'progress.json').read_bytes() == ledger_before
    assert (run_dir / 'pending.json').exists()


def test_history_discovery_does_not_open_progress_ledger(tmp_path, monkeypatch):
    env = build_env(tmp_path, monkeypatch)
    result = e2e.run_campaign(env.prepared, env.approval, cancelled=lambda: len(env.calls) == 1)
    reads = []; original = store.read_json
    def read(path):
        reads.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(store, 'read_json', read)
    history = e2e.list_campaigns(env.context['output_root'])
    assert len(history) == 1 and history[0]['run_dir'] == result['run_dir']
    assert 'status.json' in reads and 'progress.json' not in reads


def test_busy_gui_poll_and_history_do_not_read_worker_ledger(ui, monkeypatch):
    root, _, panel, env, _ = ui
    prepare_and_sign(root, panel, env)
    ready, release = threading.Event(), threading.Event()
    def hold_worker():
        ready.set()
        assert release.wait(10)
    env.after_predict = hold_worker
    panel.start_button.invoke()
    try:
        spin(root, ready.is_set)
        assert panel.busy and str(panel.history_button['state']) == 'disabled'
        with monkeypatch.context() as patch:
            patch.setattr(e2e, 'list_campaigns', lambda *_: pytest.fail('History started during run'))
            patch.setattr(e2e, 'load_campaign', lambda *_: pytest.fail('Run loaded during active worker'))
            old_read = store.read_json
            def read(path):
                assert Path(path).name != 'progress.json'
                return old_read(path)
            patch.setattr(store, 'read_json', read)
            panel.history_tree.insert('', 'end', iid='0', values=('TEST', 'PARTIAL', 1, 'TEST'))
            panel.history_tree.selection_set('0')
            panel.refresh_history()
            panel.select_history()
            root.update()  # The native poll consumes messages, not progress.json.
            assert len(env.calls) == 1
        panel.cancel_button.invoke()
    finally:
        release.set()
        spin(root, lambda: not panel.busy)
    assert panel.campaign['status'] == 'PARTIAL'
