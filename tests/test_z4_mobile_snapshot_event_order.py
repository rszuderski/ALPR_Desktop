"""Z4LIB011: real JSONL -> library_snapshot -> operation view ordering.

No model conversions, imports or mutation of existing artifacts or event records.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry
from auto_annotation_tool.gui.z4_mobile_operation_summary import summarize_operations


class FrozenClock:
    @staticmethod
    def now(tz=None):
        assert tz == timezone.utc
        return datetime(2026, 10, 14, 17, 7, 3, tzinfo=timezone.utc)

    @staticmethod
    def fromtimestamp(seconds, tz=None):
        return datetime.fromtimestamp(seconds, tz)


def archive(folder: Path) -> Path:
    package = folder / 'plate' / 'MT_TEST_INT8.alprmodel'
    package.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package, 'w') as out:
        out.writestr('manifest.json', json.dumps({
            'schema': 'alpr.model.v1', 'role': 'plate',
            'model_id': 'MT-TEST', 'variants': [{'runtime': 'tflite', 'precision': 'int8'}],
        }))
    return package


def record(root, id, operation, outcome, path=None, conversion=None):
    return registry.append_event(root, operation_id=id, operation=operation,
        outcome=outcome, artifact_path=str(path) if path else '', conversion=conversion)


def test_same_second_jsonl_library_snapshot_keeps_append_order(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, 'datetime', FrozenClock)
    managed = tmp_path / 'managed'
    model = archive(managed)
    op_id = 'OP-Z4LIB011-F4-CASE1'
    record(managed, op_id, 'SINGLE_MODEL_EXPORT', 'STARTED', model)
    record(managed, op_id, 'SINGLE_MODEL_EXPORT', 'SUCCEEDED', model)
    journal_path = registry.history_file(managed)
    saved_journal, saved_model = journal_path.read_bytes(), model.read_bytes()

    rows, events, _note = registry.library_snapshot(managed, tmp_path)
    assert [event['outcome'] for event in events] == ['STARTED', 'SUCCEEDED']
    assert events[0]['observed_utc'] == events[1]['observed_utc']
    assert summarize_operations(events)[0]['outcome'] == 'SUCCEEDED'
    assert rows[0].state == 'REGISTERED_SHA_MATCH_NOT_VERIFIED'
    assert journal_path.read_bytes() == saved_journal
    assert model.read_bytes() == saved_model


def test_same_second_conversion_summary_is_unconfirmed_not_false_conflict(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, 'datetime', FrozenClock)
    root = tmp_path / 'packages'
    conversion = {
        'role': 'MZ', 'checkpoint_sha256': 'f' * 64,
        'parent_export_operation_id': 'OP-Z4LIB011-PARENT',
    }
    record(root, 'OP-Z4LIB011-CONVERT', 'MODEL_CONVERSION', 'STARTED', conversion=conversion)
    record(root, 'OP-Z4LIB011-CONVERT', 'MODEL_CONVERSION', 'UNCONFIRMED', conversion=conversion)
    rows, events, _ = registry.library_snapshot(root, tmp_path)
    assert rows == []
    assert [event['outcome'] for event in events] == ['STARTED', 'UNCONFIRMED']
    summary = summarize_operations(events)[0]
    assert summary['outcome'] == 'UNCONFIRMED'
    assert summary['artifact_sha256'] is None


def test_success_then_new_start_stays_conflict_not_approved(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, 'datetime', FrozenClock)
    root = tmp_path / 'packages'
    model = archive(root)
    op_id = 'OP-Z4LIB011-CONFLICT'
    record(root, op_id, 'SINGLE_MODEL_EXPORT', 'SUCCEEDED', model)
    record(root, op_id, 'SINGLE_MODEL_EXPORT', 'STARTED', model)
    _rows, events, _ = registry.library_snapshot(root, tmp_path)
    summary = summarize_operations(events)[0]
    assert summary['outcome'] == 'INCONSISTENT_HISTORY'
    assert summary['artifact_sha256'] is None


def test_latest_logged_export_sha_wins_for_same_path(monkeypatch, tmp_path):
    """Two exports to one path must pin the current one, not a reversed old SHA."""
    monkeypatch.setattr(registry, 'datetime', FrozenClock)
    root = tmp_path / 'packages'
    model = archive(root)
    old = record(root, 'OP-Z4LIB011-OLDER', 'SINGLE_MODEL_EXPORT', 'SUCCEEDED', model)
    with model.open('ab') as stream:
        stream.write(b'new-version-bytes')
    new = record(root, 'OP-Z4LIB011-NEWER', 'SINGLE_MODEL_EXPORT', 'SUCCEEDED', model)
    assert old['artifact_sha256'] != new['artifact_sha256']
    before = model.read_bytes(), registry.history_file(root).read_bytes()
    rows, events, _ = registry.library_snapshot(root, tmp_path)
    assert [event['operation_id'] for event in events] == ['OP-Z4LIB011-OLDER', 'OP-Z4LIB011-NEWER']
    assert rows[0].state == 'REGISTERED_SHA_MATCH_NOT_VERIFIED'
    assert (model.read_bytes(), registry.history_file(root).read_bytes()) == before
