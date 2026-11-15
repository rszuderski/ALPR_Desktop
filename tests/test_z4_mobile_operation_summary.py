"""Z4LIB007: view-only grouping of recorded exports and conversions."""
from __future__ import annotations

import json
from pathlib import Path

from auto_annotation_tool.gui.z4_mobile_operation_summary import summarize_operations
from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary
from auto_annotation_tool.gui.z4_mobile_artifact_registry import append_event, history_file


def event(operation_id, operation, outcome, when, path="", sha=None):
    return {
        "schema": "alpr.desktop.mobile_export_event.v1", "operation_id": operation_id,
        "event_id": f"event-{when}-{outcome}", "operation": operation,
        "observed_utc": when, "outcome": outcome,
        "artifact_path": path, "artifact_sha256": sha,
    }


def test_started_succeeded_two_events_one_operation_full_evidence():
    started = event("JOB-12345678", "MODEL_CONVERSION", "STARTED", "2026-10-14T09:01:00+00:00", "C:/some.alprmodel")
    success = event("JOB-12345678", "MODEL_CONVERSION", "SUCCEEDED", "2026-10-14T09:02:00+00:00", "C:/some.alprmodel", "a" * 64)
    original = json.dumps([success, started], sort_keys=True)
    result = summarize_operations([success, started])
    assert len(result) == 1
    row = result[0]
    assert row["_event_count"] == 2
    assert row["outcome"] == "SUCCEEDED"
    assert row["artifact_sha256"] == "a" * 64
    assert [e["outcome"] for e in row["_raw_events"]] == ["STARTED", "SUCCEEDED"]
    assert "NOT_GRAPH_OR_ANDROID" in row["_authority"]
    assert json.dumps([success, started], sort_keys=True) == original


def test_started_without_terminal_not_falsely_failed_or_approved():
    row = summarize_operations([event("JOB-12345678", "SINGLE_MODEL_EXPORT", "STARTED", "2026-10-14T12:00:00+00:00")])[0]
    assert row["outcome"] == "STARTED_NO_RESULT"
    assert row["artifact_sha256"] is None


def test_failed_and_unconfirmed_keep_different_meanings():
    started = event("JOB-12345678", "MODEL_CONVERSION", "STARTED", "2026-10-14T09:00:00+00:00", "C:/partial.alprmodel")
    unconfirmed = event("JOB-12345678", "MODEL_CONVERSION", "UNCONFIRMED", "2026-10-14T10:00:00+00:00")
    fail = event("EXP-12345678", "COMPLETE_PACKAGE_EXPORT", "FAILED", "2026-10-14T10:00:00+00:00")
    rows = summarize_operations([fail, unconfirmed, started])
    assert len(rows) == 2
    converted = next(r for r in rows if r["operation"] == "MODEL_CONVERSION")
    assert converted["outcome"] == "UNCONFIRMED"
    assert converted["artifact_path"] == "C:/partial.alprmodel"
    assert converted["artifact_sha256"] is None
    assert next(r for r in rows if r["operation"] == "COMPLETE_PACKAGE_EXPORT")["outcome"] == "FAILED"


def test_conflicting_terminal_outcomes_flagged_not_approved():
    start = event("JOB-12345678", "MODEL_CONVERSION", "STARTED", "2026-10-14T09:00:00+00:00")
    fail = event("JOB-12345678", "MODEL_CONVERSION", "UNCONFIRMED", "2026-10-14T09:05:00+00:00")
    success = event("JOB-12345678", "MODEL_CONVERSION", "SUCCEEDED", "2026-10-14T09:10:00+00:00", "C:/file.alprmodel", "b" * 64)
    row = summarize_operations([success, fail, start])[0]
    assert row["outcome"] == "INCONSISTENT_HISTORY"
    assert row["artifact_sha256"] is None
    assert len(row["_raw_events"]) == 3


def test_successful_reused_operation_id_with_new_started_is_ambiguous():
    success = event("JOB-12345678", "COMPLETE_PACKAGE_EXPORT", "SUCCEEDED", "2026-10-14T09:10:00+00:00", "C:/file.alprmodel", "f" * 64)
    new_start = event("JOB-12345678", "COMPLETE_PACKAGE_EXPORT", "STARTED", "2026-10-14T09:11:00+00:00")
    assert summarize_operations([new_start, success])[0]["outcome"] == "INCONSISTENT_HISTORY"


def test_same_operation_id_different_operations_are_separate():
    a = event("JOB-12345678", "MODEL_CONVERSION", "STARTED", "2026-10-14T09:00:00+00:00")
    b = event("JOB-12345678", "COMPLETE_PACKAGE_EXPORT", "STARTED", "2026-10-14T09:00:00+00:00")
    assert len(summarize_operations([a, b])) == 2


def test_missing_id_does_not_collapse_unrelated_historical_records():
    a = event("", "MODEL_CONVERSION", "SUCCEEDED", "2026-10-14T09:00:00+00:00", "C:/one.alprmodel", "a"*64)
    b = event("", "MODEL_CONVERSION", "SUCCEEDED", "2026-10-14T09:00:00+00:00", "C:/two.alprmodel", "b"*64)
    result = summarize_operations([a, b])
    assert len(result) == 2
    assert {item["artifact_path"] for item in result} == {"C:/one.alprmodel", "C:/two.alprmodel"}


def test_two_different_success_hashes_never_silently_selected():
    a = event("JOB-12345678", "MODEL_CONVERSION", "SUCCEEDED", "2026-10-14T09:00:00+00:00", sha="a"*64)
    b = event("JOB-12345678", "MODEL_CONVERSION", "SUCCEEDED", "2026-10-14T09:01:00+00:00", sha="b"*64)
    row = summarize_operations([b,a])[0]
    assert row["outcome"] == "INCONSISTENT_HISTORY"
    assert row["artifact_sha256"] is None


class FakeVar:
    def __init__(self, initial=None):
        self.value = initial
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class FakeTree:
    def __init__(self):
        self.children = []
        self.values = {}
        self.headings = {}
    def get_children(self, parent=""):
        return tuple(self.children)
    def delete(self, iid):
        self.children.remove(iid)
        self.values.pop(iid, None)
    def insert(self, parent, where, iid, values):
        self.children.append(iid)
        self.values[iid] = values
    def move(self, iid, parent, idx):
        self.children.remove(iid)
        self.children.insert(idx,iid)
    def heading(self, key, **kwargs):
        self.headings[key] = kwargs


def fake_library(history):
    obj = object.__new__(MobileArtifactLibrary)
    obj.history = history
    obj.rows = []
    obj.by_iid = {}
    obj.events_by_iid = {}
    obj.art_tree = FakeTree()
    obj.event_tree = FakeTree()
    obj.filter_var = FakeVar("")
    obj.show_raw_events = FakeVar(False)
    obj.event_view_status = FakeVar("")
    obj.sort_state = {"artifacts": None, "events": None}
    obj.sort_labels = {
        "artifacts": {"role": "Rola", "name": "Nazwa", "variants": "Runtime / precyzja",
                      "size": "Rozmiar", "date": "Data pliku (UTC)", "state": "Stan"},
        "events": {"date": "Czas (UTC)", "operation": "Operacja", "outcome": "Wynik",
                   "file": "Plik", "sha": "SHA-256"},
    }
    return obj


def test_ui_switch_operations_and_raw_records_sort_survives():
    start = event("JOB-12345678", "MODEL_CONVERSION", "STARTED", "2026-10-14T09:00:00+00:00")
    finish = event("JOB-12345678", "MODEL_CONVERSION", "SUCCEEDED", "2026-10-14T09:05:00+00:00", "C:/model.alprmodel", "b"*64)
    other = event("JOB-87654321", "COMPLETE_PACKAGE_EXPORT", "FAILED", "2026-10-14T09:06:00+00:00")
    obj = fake_library([other, finish, start])
    obj.sort_column("events", "date")
    obj.refresh_views()
    assert len(obj.event_tree.children) == 2
    assert any("(2 zdarzeń)" in value[1] for value in obj.event_tree.values.values())
    assert obj.history_counts["export_operations"] == 2 and obj.history_counts["export_events"] == 3
    assert "Wiersze po filtrze: 2 / 2" in obj.event_view_status.get()
    obj.show_raw_events.set(True)
    obj.refresh_views()
    assert len(obj.event_tree.children) == 3
    assert all("(2 zdarzeń)" not in value[1] for value in obj.event_tree.values.values())
    assert obj.event_tree.headings["date"]["text"].endswith("↑")
    obj.show_raw_events.set(False)
    obj.filter_var.set("model.alprmodel")
    obj.refresh_views()
    assert len(obj.event_tree.children) == 1
    assert obj.events_by_iid[obj.event_tree.children[0]]["_event_count"] == 2


def test_summary_and_ui_do_not_write_jsonl_or_model_artifacts(tmp_path):
    package = tmp_path / "model.alprmodel"
    package.write_bytes(b"do-not-modify-model")
    root = tmp_path / "store"
    append_event(root, operation_id="JOB-12345678", operation="SINGLE_MODEL_EXPORT", outcome="STARTED")
    history = history_file(root)
    before_pkg, before_history = package.read_bytes(), history.read_bytes()
    events = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    obj = fake_library(events)
    obj.refresh_views()
    obj.show_raw_events.set(True)
    obj.refresh_views()
    assert package.read_bytes() == before_pkg
    assert history.read_bytes() == before_history


def test_gui_has_audit_toggle_and_preserves_raw_history():
    source = (Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_mobile_artifact_library.py").read_text(encoding="utf-8")
    assert "Pokaż każde zdarzenie eksportu" in source
    assert "summarize_operations(self.history)" in source
    assert "self.history if raw_view" in source
