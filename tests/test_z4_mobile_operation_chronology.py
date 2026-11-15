"""Z4LIB008: same-second event ordering in the append-only conversion audit."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from auto_annotation_tool.gui.z4_mobile_operation_summary import summarize_operations
from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry


STAMP = "2026-10-14T15:00:00+00:00"


def record(op_id: str, status: str, *, when: str = STAMP, digest: str | None = None):
    return {
        "operation_id": op_id,
        "operation": "MODEL_CONVERSION",
        "outcome": status,
        "observed_utc": when,
        "artifact_path": "C:/models/test.alprmodel",
        "artifact_sha256": digest,
    }


def test_success_after_started_in_same_second_is_not_conflict():
    a = record("OP-CONVERT-MT1", "STARTED")
    b = record("OP-CONVERT-MT1", "SUCCEEDED", digest="a" * 64)
    row = summarize_operations([a, b])[0]
    assert row["outcome"] == "SUCCEEDED"
    assert row["artifact_sha256"] == "a" * 64
    assert [e["outcome"] for e in row["_raw_events"]] == ["STARTED", "SUCCEEDED"]
    assert row["_event_count"] == 2


def test_failure_after_started_in_same_second_is_final_failed():
    rows = [record("OP-CONVERT-MZ1", "STARTED"),
            record("OP-CONVERT-MZ1", "UNCONFIRMED")]
    row = summarize_operations(rows)[0]
    assert row["outcome"] == "UNCONFIRMED"
    assert row["artifact_sha256"] is None
    assert [e["outcome"] for e in row["_raw_events"]] == ["STARTED", "UNCONFIRMED"]


def test_later_started_after_success_same_second_is_inconsistent():
    rows = [record("OP-CONVERT-MT2", "SUCCEEDED", digest="b" * 64),
            record("OP-CONVERT-MT2", "STARTED")]
    row = summarize_operations(rows)[0]
    assert row["outcome"] == "INCONSISTENT_HISTORY"
    assert row["artifact_sha256"] is None
    assert [e["outcome"] for e in row["_raw_events"]] == ["SUCCEEDED", "STARTED"]


def test_distinct_timestamps_still_sort_chronologically_for_legacy_input():
    a = record("OP-CONVERT-MT3", "STARTED", when="2026-10-14T13:01:00+00:00")
    b = record("OP-CONVERT-MT3", "SUCCEEDED", when="2026-10-14T13:02:00+00:00", digest="c" * 64)
    row = summarize_operations([b, a])[0]
    assert row["outcome"] == "SUCCEEDED"
    assert [e["outcome"] for e in row["_raw_events"]] == ["STARTED", "SUCCEEDED"]


def test_actual_jsonl_same_second_recording_preserves_append_order(monkeypatch, tmp_path):
    # Force identical timestamps, reproducing the registry's seconds precision.
    class FrozenDateTime:
        @staticmethod
        def now(tz=None):
            assert tz == timezone.utc
            return datetime(2026, 10, 14, 15, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(registry, "datetime", FrozenDateTime)
    root = tmp_path / "mobile_packages"
    archive = tmp_path / "result.alprmodel"
    archive.write_bytes(b"fixed-artifact")
    op = "OP-MOBILE-AUDIT-SAME-SECOND"
    registry.append_event(root, operation_id=op, operation="SINGLE_MODEL_EXPORT",
                          outcome="STARTED", artifact_path=str(archive))
    registry.append_event(root, operation_id=op, operation="SINGLE_MODEL_EXPORT",
                          outcome="SUCCEEDED", artifact_path=str(archive))
    history_path = registry.history_file(root)
    before_history, before_model = history_path.read_bytes(), archive.read_bytes()
    events, warning = registry.load_history(root)
    assert warning == ""
    assert len(events) == 2
    assert events[0]["observed_utc"] == events[1]["observed_utc"]
    row = summarize_operations(events)[0]
    assert row["outcome"] == "SUCCEEDED"
    assert row["artifact_sha256"] == registry.sha256_file(archive)
    assert [e["outcome"] for e in row["_raw_events"]] == ["STARTED", "SUCCEEDED"]
    assert history_path.read_bytes() == before_history
    assert archive.read_bytes() == before_model
