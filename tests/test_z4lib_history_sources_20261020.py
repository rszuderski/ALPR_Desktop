# -*- coding: utf-8 -*-
"""Z4 history sources: real export ledger, PC audit and unlogged disk artifacts.

No fabricated history, no mutations of legacy models or ledgers, no Android
approval inferred from stored PC PASS. GUI smoke uses real Tk when displayed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest

from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry
from auto_annotation_tool.gui.z4_mobile_artifact_validation_history import (
    read_check_events, record_desktop_check, FILENAME,
)
from auto_annotation_tool.gui.z4_mobile_history_presentation import (
    build_history_presentation, SOURCE_EXPORT, SOURCE_CHECK, SOURCE_FILE,
    history_detail_card, history_operation_label, history_outcome_label,
)


def package(root: Path, name="old.alprmodel") -> Path:
    path = root / "plate" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({
            "schema": "alpr.model.v1", "role": "plate", "model_id": "MT-TEST",
            "variants": [{"runtime": "tflite", "precision": "int8"}],
        }))
    return path


def check(root: Path, file: Path, *, passed=True):
    sha = registry.sha256_file(file) if passed else ""
    return record_desktop_check(root, path=str(file), kind="MODEL", passed=passed, sha256=sha)


def view(root: Path, repo: Path, *, raw=False):
    rows, logged, note = registry.library_snapshot(root, repo)
    observed, pc_note = read_check_events(root)
    out, counts = build_history_presentation(rows, logged, observed, raw_exports=raw)
    return out, counts, logged, note, pc_note


def test_no_log_lists_discovery_without_creating_event(tmp_path):
    root = tmp_path / "mobile"
    old = package(root)
    original = old.read_bytes()
    result, counts, logged, message, _ = view(root, tmp_path)
    assert len(result) == 1 and result[0]["_history_origin"] == SOURCE_FILE
    assert counts == {"export_operations": 0, "export_events": 0, "pc_checks": 0, "unlogged_packages": 1}
    assert result[0]["observed_utc"] == "" and result[0]["artifact_sha256"] == ""
    assert result[0]["_file_modified_utc"]
    assert "NIE jest" in history_detail_card(result[0]) and "NIE jest datą konwersji" in history_detail_card(result[0])
    assert logged == [] and "Starsze" in message
    assert old.read_bytes() == original
    assert not registry.history_file(root).exists()
    assert not (root / "_registry" / FILENAME).exists()


def test_pc_history_is_shown_in_addition_to_discovered_file(tmp_path):
    root = tmp_path / "mobile"
    p = package(root)
    recorded = check(root, p)
    result, counts, *_ = view(root, tmp_path)
    assert {r["_history_origin"] for r in result} == {SOURCE_CHECK, SOURCE_FILE}
    assert counts["export_events"] == 0 and counts["pc_checks"] == counts["unlogged_packages"] == 1
    pc = next(r for r in result if r["_history_origin"] == SOURCE_CHECK)
    assert pc["operation_id"] == "" and pc["artifact_sha256"] == recorded["sha256"]
    assert history_outcome_label(pc) == "PASS (wówczas)"
    msg = history_detail_card(pc)
    assert "nie jest wpis konwersji" in msg and "Historyczny PASS" in msg


def test_multiple_real_pc_records_are_not_collapsed(tmp_path):
    root = tmp_path / "mobile"
    p = package(root)
    check(root, p)
    check(root, p, passed=False)
    checks, note = read_check_events(root)
    result, counts, *_ = view(root, tmp_path)
    pcs = [x for x in result if x["_history_origin"] == SOURCE_CHECK]
    assert len(checks) == len(pcs) == 2 and not note
    assert {x["outcome"] for x in pcs} == {"PC_PASSED", "PC_FAILED"}
    assert pcs[0]["_history_source_index"] != pcs[1]["_history_source_index"]
    assert counts["pc_checks"] == 2


def test_export_log_remains_authority_and_does_not_create_discovery_duplicate(tmp_path):
    root = tmp_path / "mobile"
    exported = package(root, "exported.alprmodel")
    orphan = package(root, "without_log.alprmodel")
    check(root, exported)
    registry.append_event(root, operation_id="Z4REAL00001", operation="SINGLE_MODEL_EXPORT", outcome="STARTED")
    registry.append_event(root, operation_id="Z4REAL00001", operation="SINGLE_MODEL_EXPORT", outcome="SUCCEEDED", artifact_path=str(exported))
    before = [exported.read_bytes(), orphan.read_bytes(), registry.history_file(root).read_bytes(),
              (root / "_registry" / FILENAME).read_bytes()]
    summary, counts, *_ = view(root, tmp_path, raw=False)
    raw, counts_raw, *_ = view(root, tmp_path, raw=True)
    assert counts == {"export_operations": 1, "export_events": 2, "pc_checks": 1, "unlogged_packages": 1}
    assert counts_raw == counts
    assert len(summary) == 3 and len(raw) == 4
    assert [x["_history_origin"] for x in raw].count(SOURCE_EXPORT) == 2
    assert len([x for x in summary if x["_history_origin"] == SOURCE_FILE]) == 1
    assert next(x for x in summary if x["_history_origin"] == SOURCE_EXPORT)["_event_count"] == 2
    assert history_operation_label(next(x for x in summary if x["_history_origin"] == SOURCE_EXPORT)) == "Eksport modelu (2 zdarzeń)"
    after = [exported.read_bytes(), orphan.read_bytes(), registry.history_file(root).read_bytes(),
             (root / "_registry" / FILENAME).read_bytes()]
    assert after == before  # read-only union


def test_error_conversion_does_not_promote_pc_success(tmp_path):
    root = tmp_path / "mobile"
    p = package(root)
    check(root, p)
    registry.append_event(root, operation_id="Z4FAIL00001", operation="SINGLE_MODEL_EXPORT", outcome="FAILED", detail="some error")
    rows, counts, *_ = view(root, tmp_path)
    exports = [x for x in rows if x["_history_origin"] == SOURCE_EXPORT]
    assert len(exports) == 1 and exports[0]["outcome"] == "FAILED"
    assert counts["unlogged_packages"] == 1
    assert history_outcome_label(exports[0]) == "BŁĄD"


def test_pc_reader_rejects_junction_or_symlink(tmp_path):
    root = tmp_path / "mobile"
    (root / "_registry").mkdir(parents=True)
    external = tmp_path / "elsewhere"
    external.write_text("secret", encoding="utf-8")
    path = root / "_registry" / FILENAME
    try:
        path.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink unavailable")
    rows, warning = read_check_events(root)
    assert rows == [] and "bezpiecznie" in warning
    assert external.read_text(encoding="utf-8") == "secret"


def test_pc_reader_partial_or_corrupt_lines_are_not_history(tmp_path):
    root = tmp_path / "mobile"
    p = package(root)
    check(root, p)
    file = root / "_registry" / FILENAME
    with file.open("ab") as out:
        out.write(b"not-json\n")
        out.write(b'{"schema":"other"}\n')
        out.write(b'{"schema":"alpr.desktop_package_validation.v1"')  # truncated
    rows, note = read_check_events(root)
    assert len(rows) == 1 and "Pominięto 3" in note
    assert rows[0]["_ledger_index"] == 0


def test_pc_reader_rejects_oversized_history_without_claiming_completeness(tmp_path):
    root = tmp_path / "mobile"
    folder = root / "_registry"
    folder.mkdir(parents=True)
    (folder / FILENAME).write_bytes(b"x" * (6 * 1024 * 1024 + 1))
    rows, note = read_check_events(root)
    assert rows == [] and "limit" in note


def test_model_research_candidate_not_mislabeled_as_alpr_package(tmp_path):
    obj = SimpleNamespace(path="/output/research_model.tflite", kind="RESEARCH_TFLITE", modified_utc="2026-10-20")
    out, counts = build_history_presentation([obj], [], [])
    assert out == [] and counts["unlogged_packages"] == 0


def test_sources_are_sorted_and_input_list_not_modified():
    rows = [SimpleNamespace(kind="MODEL", path="/z4/a.alprmodel", state="DISCOVERED_NOT_VERIFIED",
                            modified_utc="2026-01-01T00:00:00+00:00", role="MT")]
    events = [{"operation":"SINGLE_MODEL_EXPORT", "operation_id":"Z4EV00011", "event_id":"x",
               "outcome":"FAILED", "observed_utc":"2026-02-01T00:00:00+00:00", "artifact_path":""}]
    original = json.dumps(events)
    out, counts = build_history_presentation(rows, events, [])
    assert out[0]["_history_origin"] == SOURCE_EXPORT
    assert out[1]["_history_origin"] == SOURCE_FILE
    assert json.dumps(events) == original


def test_gui_browse_real_tk_displays_three_sources_and_filter(tmp_path):
    """Run via xvfb-run on Linux or native display on Windows."""
    import tkinter as tk
    from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    base = tmp_path / "mobile_packages"
    a = package(base, "pc_unlogged.alprmodel")
    b = package(base, "recorded_export.alprmodel")
    check(base, a)
    registry.append_event(base, operation_id="Z4GUITEST001", operation="SINGLE_MODEL_EXPORT", outcome="STARTED")
    registry.append_event(base, operation_id="Z4GUITEST001", operation="SINGLE_MODEL_EXPORT", outcome="SUCCEEDED", artifact_path=str(b))
    window = None
    try:
        window = MobileArtifactLibrary(root, packages_root=base, repo_root=tmp_path)
        deadline = time.monotonic() + 8
        while window.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.015)
        root.update()
        assert not window.busy and len(window.rows) == 2
        window.notebook.select(window.events_tab)
        root.update()
        origins = [e["_history_origin"] for e in window.events_by_iid.values()]
        assert sorted(origins) == sorted([SOURCE_CHECK, SOURCE_EXPORT, SOURCE_FILE])
        assert window.history_counts["pc_checks"] == 1
        assert "1  Kontrole PC" == window._source_cards["pc_checks"][1].cget("text")
        # Each source has an explicit, non-fabricated authority and date.
        for iid, record in list(window.events_by_iid.items()):
            window.event_tree.selection_set(iid)
            window.event_selected()
            dialog = window.show_details()
            details = dialog.summary.get("1.0", "end")
            dialog.close()
            if record["_history_origin"] == SOURCE_FILE:
                assert "Data modyfikacji PLIKU" in details and "NIE jest datą konwersji" in details
            elif record["_history_origin"] == SOURCE_CHECK:
                assert "Historyczny PASS" in details
            else:
                assert "Eksport modelu" in details and "REJESTR EKSPORTU" in details
        window.history_source_filter.set("Kontrole PC")
        window.refresh_views()
        assert len(window.events_by_iid) == 1
        assert {x["_history_origin"] for x in window.events_by_iid.values()} == {SOURCE_CHECK}
        window.history_source_filter.set("Wszystkie źródła")
        window.show_raw_events.set(True)
        window.refresh_views()
        assert len(window.events_by_iid) == 4  # 2 raw exports, 1 check, 1 disk
        # Filter must not alter audit records.
        assert len(registry.load_history(base)[0]) == 2
        assert len(read_check_events(base)[0]) == 1
    finally:
        if window is not None:
            window.close()
        root.destroy()


def test_gui_new_pc_validation_appears_without_manual_rescan(tmp_path):
    """The exact user flow: a completed PC check becomes a history row immediately."""
    import tkinter as tk
    from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    base = tmp_path / "packages"
    p = package(base, "user_mt_int8.alprmodel")
    lib = None
    try:
        lib = MobileArtifactLibrary(root, packages_root=base, repo_root=tmp_path)
        deadline = time.monotonic() + 8
        while lib.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.015)
        root.update()
        assert lib._desktop_check_events == []
        assert lib.history == []
        assert len(lib.events_by_iid) == 1  # Unlogged disk artifact
        lib.notebook.select(lib.packages_tab)
        root.update()
        iid = next(iter(lib.by_iid))
        lib.art_tree.selection_set(iid)
        root.update()
        kind = lib.by_iid[iid].kind
        entry = check(base, p)
        lib._verification_active = {"token": 23, "path": str(p.resolve()), "kind": kind}
        lib.busy = True
        lib.pipe.put(("verified", {"token": 23, "path": str(p.resolve()), "kind": kind,
                                  "ok": True, "sha256": entry["sha256"],
                                  "history_record": entry, "history_warning": ""}))
        lib.poll()
        assert len(lib._desktop_check_events) == 1
        assert len(lib.history) == 0  # Crucial: not rewritten as export.
        assert len(lib.events_by_iid) == 2  # PC entry plus unlogged file
        assert any(e["_history_origin"] == SOURCE_CHECK for e in lib.events_by_iid.values())
        assert lib._details_dialog is None  # No modal after background validation.
        dialog = lib.show_details()
        assert "Kontrola PC zaliczona" in dialog.summary.get("1.0", "end")
        dialog.close()
        assert registry.load_history(base)[0] == []
        assert len(read_check_events(base)[0]) == 1
    finally:
        if lib is not None:
            lib.close()
        root.destroy()
