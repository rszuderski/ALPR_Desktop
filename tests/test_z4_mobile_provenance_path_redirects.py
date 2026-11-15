#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB024: no redirected MZ research source or mobile JSONL journal."""
from __future__ import annotations

from pathlib import Path
from unittest import mock
import json
import pytest

from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry


def _symlink_or_skip(link: Path, source: Path, *, directory: bool = False):
    try:
        link.symlink_to(source, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks unavailable on this Windows account: {exc}")


def test_research_tflite_file_symlink_not_indexed(tmp_path):
    repo = tmp_path / "repo"
    raw = repo / registry.RESEARCH_REL
    raw.parent.mkdir(parents=True)
    outside = tmp_path / "outside.tflite"
    outside.write_bytes(b"untrusted research candidate")
    _symlink_or_skip(raw, outside)
    assert registry.research_candidate(repo) is None


def test_research_parent_directory_symlink_not_indexed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "somewhere_else"
    file = outside / "mz_day_int8_raw_probe" / registry.RESEARCH_REL.name
    file.parent.mkdir(parents=True)
    file.write_bytes(b"untrusted research candidate")
    _symlink_or_skip(repo / "output", outside, directory=True)
    assert registry.research_candidate(repo) is None


def test_research_normal_file_remains_visible_but_unapproved(tmp_path):
    repo = tmp_path / "repo"
    raw = repo / registry.RESEARCH_REL
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"not a frozen research model")
    row = registry.research_candidate(repo)
    assert row is not None
    assert row.kind == "RESEARCH_TFLITE"
    assert row.state == "RESEARCH_SHA_MISMATCH"


def test_registry_parent_redirect_blocks_append(tmp_path):
    root = tmp_path / "managed"
    root.mkdir()
    outside = tmp_path / "external"
    outside.mkdir()
    _symlink_or_skip(root / "_registry", outside, directory=True)
    with pytest.raises(ValueError, match="_registry"):
        registry.append_event(root, operation_id="Z4-20261014-REDIRECT",
                              operation="COMPLETE_PACKAGE_EXPORT", outcome="STARTED")
    assert not (outside / "mobile_export_events.v1.jsonl").exists()


def test_registry_parent_redirect_blocks_history_read(tmp_path):
    root = tmp_path / "managed"
    root.mkdir()
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "mobile_export_events.v1.jsonl").write_text(json.dumps({
        "schema": registry.EVENT_SCHEMA,
        "event_id": "x", "operation_id": "Z4-20261014-X",
        "outcome": "SUCCEEDED", "artifact_path": "forged.alprmodel"
    }) + "\n", encoding="utf-8")
    _symlink_or_skip(root / "_registry", outside, directory=True)
    entries, note = registry.load_history(root)
    assert entries == []
    assert "przekierowuje" in note


def test_fake_junction_is_treated_as_redirect(tmp_path):
    """Windows junctions are not always returned by Path.is_symlink()."""
    class FakeJunction:
        def is_symlink(self):
            return False
        def is_junction(self):
            return True
    assert registry._redirecting_path(FakeJunction())


def test_journal_normal_roundtrip_unaffected(tmp_path):
    root = tmp_path / "managed"
    evt = registry.append_event(root, operation_id="Z4-20261014-NORMAL",
                                operation="COMPLETE_PACKAGE_EXPORT", outcome="STARTED")
    events, warning = registry.load_history(root)
    assert warning == ""
    assert len(events) == 1 and events[0]["event_id"] == evt["event_id"]


def test_redirect_probe_errors_fail_closed():
    class BrokenPath:
        def is_symlink(self):
            raise OSError("probe unavailable")
    assert registry._redirecting_path(BrokenPath())
