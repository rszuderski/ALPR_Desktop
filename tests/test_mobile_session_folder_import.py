"""Native folder import on synthetic archives and private stores only."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import queue
import threading
import time
import zipfile

import pytest

from auto_annotation_tool.ranking import mobile_package_experiments as core
from auto_annotation_tool.gui import z4_mobile_report_browser as gui
from test_z4_akt89k_import_preferences import (
    settings, browser_for_import, picker_host, run_picker, widgets,
)


def archive(path, variant="R0", repeat=1, block="A", *, sid=None, series=core.F5A_IMPORT_SERIES,
            scenario=core.F5A_IMPORT_SCENARIO, roles=True, edit=None):
    sid = sid or f"synthetic-{variant}-{repeat}-{block}"
    identity = dict(series_id=series, scenario_id=scenario, variant=variant, replicate_index=repeat)
    session = dict(identity, schema="alpr.mobile_research_session.v1", session_id=sid,
                   experiment_type="roi_budget", mt_detection_roles_complete=roles,
                   execution=dict(roi_budget_policy={"R0": "r0_full_frame", "R1": "r1_one_roi", "R2": "r2_two_roi"}[variant]))
    report = dict(schema=core.MOBILE_BENCHMARK_REPORT_SCHEMA, report_id=f"report-{sid}",
                  measured_at="2026-01-01T00:00:00Z", package_id="synthetic", variant_id="fp32",
                  experiment=dict(identity, type="roi_budget", notes=f"BLOK_{block} · 10 zdjęć × 30 s",
                                  session=dict(identity, id=sid, notes=f"BLOK_{block}")),
                  research_collection=dict(session))
    data = {"session.json": session, "report.json": report}
    if edit:
        edit(data)
    entries = {n: json.dumps(v, ensure_ascii=False).encode("utf-8") for n, v in data.items()}
    manifest = dict(schema=core.MOBILE_RESEARCH_BUNDLE_SCHEMA, report_id=report["report_id"],
                    mt_detection_roles_complete=roles,
                    entry_sha256={n: hashlib.sha256(b).hexdigest() for n, b in entries.items()})
    entries["manifest.json"] = json.dumps(manifest).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in entries.items():
            z.writestr(n, b)
    return path


def rewrite(path, edit):
    with zipfile.ZipFile(path) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    edit(entries)
    with zipfile.ZipFile(path, "w") as z:
        for n, b in entries.items():
            z.writestr(n, b)


def plan(root, store=None, **kwargs):
    return core.plan_mobile_session_import(root, store.reports if store else (), **kwargs)


def pump(root, condition, seconds=12):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        root.update()
        time.sleep(.005)
    root.update()
    assert condition(), "Tk test condition timed out"


@pytest.fixture
def flow_gui(tmp_path, settings):
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    dialog = tk.Toplevel(root); dialog.geometry("1040x740")
    browser = picker_host(root)
    browser.store = core.MobilePackageExperimentStore(tmp_path / "flow-store")
    selection = tk.Frame(dialog); selection.pack(fill=tk.BOTH, expand=True)
    errors = []; root.report_callback_exception = lambda *args: errors.append(args)
    def close():
        if flow.request_close(): dialog.destroy()
    flow = gui._MobileFolderImportFlow(browser, dialog, selection, close)
    try:
        yield root, browser, dialog, flow
        assert not errors, errors
    finally:
        flow.cancel_event.set()
        root.destroy()


def test_recursive_36_order_idempotency_sha_and_native_reopening(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for index, (variant, repeat, block) in enumerate(reversed(core.F5A_IMPORT_COMBINATIONS)):
        archive(source / variant / core.F5A_IMPORT_SCENARIO / repeat / f"unrelated-{index}.alprsession",
                variant, int(repeat[1:]), block, roles=block != "C")
    (source / "ignore.zip").write_bytes(b"not an alprsession")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Archives must not be extracted")
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    store = core.MobilePackageExperimentStore(tmp_path / "catalog")
    progress = []
    first = plan(source, store, progress=lambda *args: progress.append(args),
                 prepare_report=gui._compact_mobile_report_for_store)
    assert not store.file_path.exists() and not store.reports
    assert len(first.items) == first.count("NEW") == 36 and first.missing == ()
    assert tuple(i.combination for i in first.items) == core.F5A_IMPORT_COMBINATIONS
    assert progress[-1][:2] == (36, 36)
    assert sum(i.roles_incomplete for i in first.items) == 9
    store.import_mobile_session_plan(first)
    assert first.count("IMPORTED") == len(store.reports) == 36
    frozen = store.file_path.read_bytes()
    reloaded = core.MobilePackageExperimentStore(store.root_dir)
    second = plan(source, reloaded)
    assert second.count("EXISTING") == 36 and second.count("NEW") == 0 and not second.missing
    reloaded.import_mobile_session_plan(second)
    assert reloaded.file_path.read_bytes() == frozen and len(reloaded.reports) == 36
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from auto_annotation_tool.gui.mobile_report_source_recovery import restore_source
    for item in first.items:
        stored = next(r for r in reloaded.reports if r.source_archive_sha256 == item.sha256)
        assert restore_source(stored, stored.source_path).ok
        reopened = core.read_mobile_report_bundle(Path(item.report.source_path))
        assert reopened.validation.ok and reopened.source_archive_sha256 == item.sha256
        assert item.sha256 == hashlib.sha256(before[item.path][0]).hexdigest()
        review = MobileReviewSession.open(item.path, sidecar_path=tmp_path / "reviews" / f"{item.session_id}.json")
        assert review.session_id == item.session_id
        assert item.report.raw["desktop_ingest"]["mt_detection_roles_complete_sources"]["session.json"] == (item.combination[2] != "C")
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()}


def test_bad_archive_foreign_scenario_pilot_and_missing_block_continue(tmp_path):
    source = tmp_path / "source"
    archive(source / "good.alprsession")
    archive(source / "foreign.alprsession", scenario="F4", sid="other-scenario")
    archive(source / "pilot.alprsession", series=core.F5A_IMPORT_SERIES + "-PILOT", sid="pilot")
    archive(source / "f4.alprsession", series="F4-OFFICIAL", sid="f4")
    (source / "broken.alprsession").write_bytes(b"broken ZIP")
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    result = plan(source, store)
    assert len(result.eligible_items) == result.count("INVALID") == 1
    assert len(result.groups) == 4 and result.count("NEW") == 4 and len(result.missing) == 35
    assert ("R0", "r01", "B") in result.missing and ("R0", "r01", "A") not in result.missing
    store.import_mobile_session_plan(result)
    assert result.count("IMPORTED") == len(store.reports) == 1
    assert "broken.alprsession" in result.summary(completed=True)


def test_exact_missing_block_in_otherwise_complete_tree(tmp_path):
    missing = ("R1", "r02", "C")
    for variant, repeat, block in core.F5A_IMPORT_COMBINATIONS:
        if (variant, repeat, block) != missing:
            archive(tmp_path / variant / core.F5A_IMPORT_SCENARIO / repeat / f"{block}.alprsession",
                    variant, int(repeat[1:]), block)
    result = plan(tmp_path)
    assert result.count("NEW") == 35 and result.missing == (missing,)


@pytest.mark.parametrize("damage", ["session_absent", "session_id", "session_schema", "hash", "manifest_absent",
                                    "hash_missing", "manifest_report_id", "unknown_manifest", "block_missing",
                                    "block_conflict", "variant_conflict", "repeat", "json_disguised"])
def test_rejects_invalid_identity_or_manifest_not_just_filename(tmp_path, damage):
    def edit(data):
        s, r = data["session.json"], data["report.json"]
        if damage == "session_absent": del data["session.json"]
        if damage == "session_id": s["session_id"] = "wrong"
        if damage == "session_schema": s["schema"] = "foreign.v1"
        if damage == "block_missing":
            r["experiment"].pop("notes"); r["experiment"]["session"].pop("notes")
        if damage == "block_conflict": r["experiment"]["session"]["notes"] = "BLOK_B"
        if damage == "variant_conflict": s["variant"] = "R2"
        if damage == "repeat": s["replicate_index"] = 4
    p = archive(tmp_path / "F5A_R0_r01_BLOK_A.alprsession", edit=edit)
    def break_entries(entries):
        if damage == "manifest_absent":
            del entries["manifest.json"]
            return
        m = json.loads(entries["manifest.json"])
        if damage == "hash": m["entry_sha256"]["session.json"] = "0" * 64
        if damage == "hash_missing": del m["entry_sha256"]["session.json"]
        if damage == "manifest_report_id": m["report_id"] = "wrong"
        if damage == "unknown_manifest": m["schema"] = "foreign.v1"
        entries["manifest.json"] = json.dumps(m).encode()
    rewrite(p, break_entries)
    if damage == "json_disguised": p.write_text('{"report_id":"fake"}', encoding="utf-8")
    result = plan(tmp_path)
    status = ("IDENTITY_CONFLICT" if damage in {"session_id", "manifest_report_id"} else
              "CONFLICT" if damage in {"block_missing", "block_conflict", "variant_conflict", "repeat"} else "INVALID")
    assert result.count(status) == 1 and len(result.missing) == (0 if damage == "json_disguised" else 36)


def test_duplicates_and_existing_review_are_preserved(tmp_path):
    p = archive(tmp_path / "source" / "one.alprsession")
    alias = p.parent / "nested" / "two.ALPRSESSION"
    alias.parent.mkdir(); alias.write_bytes(p.read_bytes())
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    result = plan(p.parent, store)
    assert result.count("NEW") == result.count("DUPLICATE") == 1
    store.import_mobile_session_plan(result)
    original = store.reports[0]
    store.reports[0] = replace(original, raw=dict(original.raw, human_review={"test": "keep"}))
    store.save()
    again = plan(p.parent, store)
    assert again.count("EXISTING") == again.count("DUPLICATE") == 1
    store.import_mobile_session_plan(again)
    assert store.reports[0].raw["human_review"] == {"test": "keep"}
    assert store.reports[0].source_path == original.source_path


@pytest.mark.parametrize("conflict", ["same_session", "same_combination"])
def test_ambiguous_sources_never_choose_a_winner(tmp_path, conflict):
    archive(tmp_path / "one.alprsession")
    archive(tmp_path / "two.alprsession", sid="synthetic-R0-1-A" if conflict == "same_session" else "another",
            block="B" if conflict == "same_session" else "A")
    result = plan(tmp_path)
    assert result.count("IDENTITY_CONFLICT") == 2 and result.count("NEW") == 0 and len(result.missing) == 36


def test_changed_sha_after_preview_and_store_failure_continue(tmp_path, monkeypatch):
    source = tmp_path / "source"
    paths = [archive(source / f"{b}.alprsession", block=b) for b in "ABCD"]
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    result = plan(source, store)
    paths[0].write_bytes(b"source changed")
    original = store.save
    calls = []
    def save():
        calls.append(True)
        if len(calls) == 1: raise OSError("synthetic disk error")
        original()
    monkeypatch.setattr(store, "save", save)
    store.import_mobile_session_plan(result)
    assert result.count("ERROR") == result.count("IMPORTED") == 2
    assert len(core.MobilePackageExperimentStore(store.root_dir).reports) == 2
    assert "SHA-256" in result.items[0].details[0] and "synthetic disk error" in result.items[1].details[0]


def test_existing_session_with_different_sha_rejected(tmp_path):
    source = tmp_path / "source"
    p = archive(source / "one.alprsession")
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    store.import_mobile_session_plan(plan(source))
    archive(p, roles=False)
    result = plan(source, store)
    assert result.count("IDENTITY_CONFLICT") == 1 and result.count("NEW") == 0


def test_store_rechecked_after_preview_and_existing_combination_protected(tmp_path):
    first = archive(tmp_path / "source" / "one.alprsession")
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    preview = plan(first.parent)
    store.import_mobile_session_plan(plan(first.parent))
    store.import_mobile_session_plan(preview)
    assert preview.count("EXISTING") == 1 and len(store.reports) == 1
    archive(tmp_path / "other" / "same_slot.alprsession", sid="new-session")
    conflicting = plan(tmp_path / "other", store)
    assert conflicting.count("IDENTITY_CONFLICT") == 1
    assert "kombinacji F5A" in conflicting.items[0].reason


def test_scan_directory_failure_reported_without_discarding_other_files(tmp_path, monkeypatch):
    p = archive(tmp_path / "source" / "one.alprsession")
    def walk(root, *, onerror, followlinks):
        onerror(PermissionError("synthetic inaccessible subdirectory"))
        yield str(p.parent), [], [p.name]
    monkeypatch.setattr(core.os, "walk", walk)
    result = plan(p.parent)
    assert result.count("NEW") == 1
    assert len(result.scan_errors) == 1 and "Brak dostępu" in result.scan_errors[0]
    assert "synthetic inaccessible subdirectory" in result.scan_errors[0]


def test_changing_source_during_validation_is_rejected(tmp_path, monkeypatch):
    p = archive(tmp_path / "one.alprsession")
    original = core.read_mobile_report_bundles
    def read(path):
        result = original(path)
        path.write_bytes(b"changed after validator read")
        return result
    monkeypatch.setattr(core, "read_mobile_report_bundles", read)
    result = plan(tmp_path)
    assert result.count("ERROR") == 1 and "podczas odczytu" in result.items[0].reason


def test_junction_directory_is_not_followed(tmp_path, monkeypatch):
    p = archive(tmp_path / "source" / "one.alprsession")
    visited = []
    def walk(root, *, onerror, followlinks):
        dirs = ["linked"]
        yield str(root), dirs, [p.name]
        visited.extend(dirs)
    monkeypatch.setattr(core.os, "walk", walk)
    monkeypatch.setattr(core.os.path, "isjunction", lambda path: Path(path).name == "linked", raising=False)
    assert plan(p.parent).count("NEW") == 1 and visited == []


def test_generic_folder_mode_and_empty_scan(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    assert len(plan(root).missing) == 0
    archive(root / "foreign.alprsession", series="F4-OFFICIAL")
    generic = plan(root, f5a=False)
    assert generic.count("NEW") == 1 and not generic.missing


def test_legacy_single_multiple_json_and_alprsession_still_import(tmp_path):
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    single = tmp_path / "single.json"
    single.write_text(json.dumps({"report_id": "single", "measured_at": "2026-01-01"}))
    assert len(store.import_mobile_report_file(single)) == 1
    multi = tmp_path / "many.json"
    multi.write_text(json.dumps({"reports": [{"report_id": "a"}, {"report_id": "b"}]}))
    assert len(store.import_mobile_report_file(multi)) == 2
    p = archive(tmp_path / "legacy.alprsession", edit=lambda data: data["report.json"].pop("measured_at"))
    store.import_mobile_report_file(p)
    before = len(store.reports)
    store.import_mobile_report_file(p)
    assert len(store.reports) == before == 4
    zip_alias = tmp_path / "same-session.zip"
    zip_alias.write_bytes(p.read_bytes())
    store.import_mobile_report_file(zip_alias)
    assert len(store.reports) == before
    disguised = tmp_path / "many.alprsession"
    disguised.write_text(json.dumps({"reports": [{"report_id": "disguised-a"}, {"report_id": "disguised-b"}]}))
    assert len(store.import_mobile_report_file(disguised)) == 2
    assert len(store.reports) == 6


def test_missing_date_remains_unknown_and_native_samples_reopen_after_reload(tmp_path, monkeypatch):
    from PIL import Image
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from auto_annotation_tool.gui.mobile_report_source_recovery import restore_source
    p = archive(tmp_path / "source" / "sample.alprsession",
                edit=lambda data: data["report.json"].pop("measured_at"))
    def add_sample(entries):
        image = io.BytesIO(); Image.new("RGB", (64, 32), "white").save(image, format="JPEG")
        entries["samples/index.csv"] = b"capture_id,subject_key,session_id,prediction\nc1,synthetic-R0-1-A/sg-1/entity-1,synthetic-R0-1-A,TEST\n"
        entries["samples/crops/c1.jpg"] = image.getvalue()
        manifest = json.loads(entries["manifest.json"])
        manifest["entry_sha256"] = {n: hashlib.sha256(b).hexdigest() for n, b in entries.items() if n != "manifest.json"}
        entries["manifest.json"] = json.dumps(manifest).encode()
    rewrite(p, add_sample)
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    store.import_mobile_session_plan(plan(p.parent))
    monkeypatch.setattr(core, "_utc_now_iso", lambda: "2030-01-01T00:00:00Z")
    stored = core.MobilePackageExperimentStore(store.root_dir).reports[0]
    assert stored.measured_at == ""
    restored = restore_source(stored, stored.source_path)
    assert restored.ok
    review = MobileReviewSession.open(p, sidecar_path=tmp_path / "private-review.json")
    assert list(review.samples) == ["c1"]
    assert core.read_mobile_sample_image(restored.bundle, "samples/crops/c1.jpg").size == (64, 32)


def test_folder_picker_honors_typed_root_over_old_selection(tmp_path, settings):
    import tkinter as tk
    source, typed = tmp_path / "source", tmp_path / "typed"
    source.mkdir(); typed.mkdir(); (source / "selected").mkdir()
    def accept(dialog):
        listing = next(w for w in widgets(dialog) if isinstance(w, tk.Listbox))
        selected = next(i for i, text in enumerate(listing.get(0, "end")) if "selected" in text)
        listing.selection_set(selected)
        entry = next(w for w in widgets(dialog) if isinstance(w, tk.Entry))
        entry.delete(0, "end"); entry.insert(0, str(typed))
        next(w for w in widgets(dialog) if w.winfo_class() == "TButton"
             and w.cget("text") == "Sprawdź folder z podfolderami").invoke()
        flow = dialog._mobile_import_flow
        pump(dialog, lambda: flow.phase == "result")
        assert flow.plan.root == typed
    assert run_picker(source, accept) == ()


def test_cancellation_before_commit_does_not_write(tmp_path):
    archive(tmp_path / "source" / "one.alprsession")
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    result = plan(tmp_path / "source")
    store.import_mobile_session_plan(result, cancelled=lambda: True)
    assert result.cancelled and not store.file_path.exists() and not store.reports
    assert len(result.missing) == 36


def test_T05_T08_T09_T11_32_pass_4_policy_conflicts_keep_all_facts(tmp_path):
    for number, (variant, repeat, block) in enumerate(core.F5A_IMPORT_COMBINATIONS):
        def wrong_policy(data):
            if variant == "R1" and repeat == "r02":
                data["session.json"]["execution"]["roi_budget_policy"] = "r2_two_roi"
        archive(tmp_path / "source" / f"arbitrary-{number % 5}" / f"file-{number}.alprsession",
                variant, int(repeat[1:]), block, edit=wrong_policy)
    before = {p: p.read_bytes() for p in (tmp_path / "source").rglob("*.alprsession")}
    result = plan(tmp_path / "source")
    assert len(result.items) == sum(i.archive_ok is True for i in result.items) == 36
    assert result.count("NEW") == 32 and result.count("CONFLICT") == 4
    assert result.count("INVALID") == 0 and result.missing_files == ()
    assert result.missing == tuple(("R1", "r02", b) for b in "ABCD")
    conflicts = [i for i in result.items if i.status == "CONFLICT"]
    for item in conflicts:
        assert item.session_id and item.combination[0:2] == ("R1", "r02")
        assert item.observations["roi_budget_policy"] == ("r2_two_roi",)
        assert item.codes == ("VARIANT_POLICY_CONFLICT",)
        assert item.reason == "Niezgodna strategia ROI" and item.profile_ok is False
        assert "R2 — maksymalnie 2 ROI" in "\n".join(item.details)
        assert "R1 — maksymalnie 1 ROI" in "\n".join(item.details)
    summary = result.summary()
    for text in ["Wszystkie znalezione archiwa: 36", "Zgodne z protokołem F5A: 32", "Niezgodne z protokołem F5A: 4",
                 "Brakujących plików oryginalnych F5A: 0", "Brakujących poprawnych pomiarów F5A: 4"]:
        assert text in summary
    assert "N/A" not in summary
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    store.import_mobile_session_plan(result)
    assert len(store.reports) == 32 and result.count("IMPORTED") == 32
    assert all(r.raw["research_collection"]["execution"]["roi_budget_policy"] != "r2_two_roi"
               for r in store.reports if r.raw["experiment"]["variant"] == "R1")
    broken = tmp_path / "source" / "broken.alprsession"; broken.write_bytes(b"bad ZIP")
    with_bad = plan(tmp_path / "source", store)
    assert with_bad.count("CONFLICT") == 4 and with_bad.count("INVALID") == 1
    assert "Brakujących plików oryginalnych F5A: nie można ustalić" in with_bad.summary()
    assert all(p.read_bytes() == b for p, b in before.items())


def test_T06_general_is_default_no_folder_name_or_f5a_requirement(tmp_path):
    archive(tmp_path / "arbitrary" / "branch" / "foreign.alprsession", series="OTHER", scenario="ANY")
    result = core.plan_mobile_session_import(tmp_path)
    assert not result.f5a and result.count("NEW") == 1 and not result.missing
    assert result.items[0].profile_ok is None
    assert "bez przypisania do F5A" in result.summary()
    assert "Brakujących plików oryginalnych" not in result.summary()


def test_T12_incomplete_mt_counts_are_source_based_warnings(tmp_path, monkeypatch):
    archive(tmp_path / "one.alprsession", roles=False)
    original = core.read_mobile_report_bundles
    def read(path):
        bundle = original(path)[0]
        roles = [dict(postprocess_status="NOT_REACHED", cancel_reason=reason)
                 for reason, n in (("scene_superseded", 45), ("session_stopped", 12)) for _ in range(n)]
        roles.append(dict(postprocess_status="COMPLETED", cancel_reason="scene_superseded"))
        return (replace(bundle, scene_evidence={"roles": roles}),)
    monkeypatch.setattr(core, "read_mobile_report_bundles", read)
    result = plan(tmp_path)
    assert result.count("NEW") == 1 and result.items[0].archive_ok
    assert result.items[0].mt_interruptions == {"scene_superseded": 45, "session_stopped": 12}
    for text in ["postprocessingiem: 57", "scene_superseded: 45", "session_stopped: 12",
                 "Nie wszystkie wywołania MT zakończyły postprocessing", "nie błąd ZIP/SHA"]:
        assert text in result.summary()


def test_T04_highlighted_folder_never_silently_changes_scan_root(tmp_path, settings):
    import tkinter as tk
    root = tmp_path / "root"; root.mkdir(); (root / "highlighted").mkdir()
    def accept(dialog):
        listing = next(w for w in widgets(dialog) if isinstance(w, tk.Listbox))
        i = next(i for i, text in enumerate(listing.get(0, "end")) if "highlighted" in text)
        listing.selection_set(i)
        next(w for w in widgets(dialog) if w.winfo_class() == "TButton" and w.cget("text") == "Sprawdź folder z podfolderami").invoke()
        flow = dialog._mobile_import_flow
        pump(dialog, lambda: flow.phase == "result")
        assert flow.plan.root == root
    assert run_picker(root, accept) == ()


def test_T13_T14_file_reimport_also_preserves_review_path_and_sha(tmp_path, settings, monkeypatch):
    p = archive(tmp_path / "one.alprsession")
    b = browser_for_import(tmp_path, monkeypatch, [p])
    b.import_report()
    record = b.store.reports[0]
    b.store.reports[0] = replace(record, raw=dict(record.raw, human_review={"operator_decision": "keep"}))
    b.store.save()
    before = b.store.file_path.read_bytes()
    b.import_report()
    assert b.store.file_path.read_bytes() == before
    assert b.store.reports[0].raw["human_review"] == {"operator_decision": "keep"}
    alias = tmp_path / "elsewhere" / "same.alprsession"; alias.parent.mkdir(); alias.write_bytes(p.read_bytes())
    b.store.import_mobile_report_file(alias)
    assert b.store.file_path.read_bytes() == before
    assert b.store.reports[0].source_path == record.source_path and b.store.reports[0].source_archive_sha256 == record.source_archive_sha256


def test_T17_cancel_running_scan_without_closing_browser(tmp_path):
    for block in "AB": archive(tmp_path / f"{block}.alprsession", block=block)
    host = object.__new__(gui.MobileReportBrowser)
    host._import_cancel_requested = False
    messages = []; host._set_status = lambda text, tone: messages.append(text)
    def progress(done, total, path):
        if done == 1: host._cancel_mobile_import()
    result = plan(tmp_path, progress=progress, cancelled=lambda: host._import_cancel_requested)
    assert result.cancelled and len(result.items) == 1
    assert "Anulowanie" in messages[-1]


@pytest.mark.parametrize("case", ["missing", "permission", "drive", "empty", "no_archives", "long_path"])
def test_T01_folder_errors_empty_and_long_windows_paths(tmp_path, monkeypatch, case):
    root = tmp_path / "folder"; root.mkdir()
    if case == "missing": root = root / "not_here"
    if case in {"permission", "drive"}:
        original = core.os.scandir
        def scandir(path):
            if Path(path) == root:
                if case == "permission": raise PermissionError("synthetic denied")
                raise FileNotFoundError("synthetic disconnected drive")
            return original(path)
        monkeypatch.setattr(core.os, "scandir", scandir)
    if case == "no_archives": (root / "unrelated.txt").write_text("hello")
    if case == "long_path":
        root = root / ("a" * 90) / ("b" * 90) / ("c" * 90)
        archive(root / "one.alprsession")
        assert len(str(root)) > 260
    if case in {"missing", "permission", "drive"}:
        with pytest.raises(ValueError, match="Brak dostępu|dysk jest niedostępny"):
            core.plan_mobile_session_import(root)
    else:
        result = core.plan_mobile_session_import(root)
        assert result.count("NEW") == (1 if case == "long_path" else 0)
        if case != "long_path": assert "Nie znaleziono plików .alprsession" in result.summary()


def ux02_tree(root, *, pilots=True):
    paths = []
    for n, (variant, repeat, block) in enumerate(core.F5A_IMPORT_COMBINATIONS):
        def policy(data):
            if variant == "R1" and repeat == "r02":
                data["session.json"]["execution"]["roi_budget_policy"] = "r2_two_roi"
        paths.append(archive(root / f"branch-{n % 5}" / f"archive-{n}.alprsession",
                             variant, int(repeat[1:]), block, roles=n >= 23, edit=policy))
    if pilots:
        for n, name in enumerate(("pilot-old", "pilot-new")):
            paths.append(archive(root / "unrelated" / f"{name}.alprsession", sid=name,
                                 series=core.F5A_IMPORT_SERIES + "-PILOT", roles=n == 0))
    return paths


@pytest.mark.parametrize("compatibility_flag", [None, False, True])
def test_UX02_T02_T04_T08_auto_f5a_no_generic_bypass(tmp_path, compatibility_flag):
    ux02_tree(tmp_path, pilots=False)
    result = core.plan_mobile_session_import(tmp_path, f5a=compatibility_flag)
    assert result.f5a and result.selected_group == core.F5A_IMPORT_GROUP
    assert len(result.eligible_items) == 32 and result.count("CONFLICT") == 4
    assert result.import_caption == "Importuj 32 zgodne sesje"
    assert all(i.codes == ("VARIANT_POLICY_CONFLICT",) and i.archive_ok for i in result.items if i.status == "CONFLICT")


@pytest.mark.parametrize("series,scenario", [
    (core.F5A_IMPORT_SERIES + "-PILOT", core.F5A_IMPORT_SCENARIO),
    (core.F5A_IMPORT_SERIES + "-EXTRA", core.F5A_IMPORT_SCENARIO),
    (core.F5A_IMPORT_SERIES.lower(), core.F5A_IMPORT_SCENARIO),
    (core.F5A_IMPORT_SERIES + " ", core.F5A_IMPORT_SCENARIO),
    (core.F5A_IMPORT_SERIES, "OTHER"),
])
def test_UX02_T03_exact_metadata_only_never_filename_prefix(tmp_path, series, scenario):
    archive(tmp_path / core.F5A_IMPORT_SERIES / "F5A_R1_r02_BLOK_A.alprsession", series=series, scenario=scenario)
    result = plan(tmp_path)
    assert not result.f5a and not result.items[0].is_f5a
    assert result.items[0].profile_ok is None and len(result.eligible_items) == 1


def test_UX02_T05_T06_T07_T23_mixed_root_selects_32_preserves_existing_pilot(tmp_path):
    paths = ux02_tree(tmp_path / "source")
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    original = store.import_mobile_report_file(paths[-2])[0]
    store.reports[0] = replace(original, raw=dict(original.raw, human_review={"approved": True}))
    store.save()
    before = store.file_path.read_bytes()
    result = plan(tmp_path / "source", store)
    assert len(result.items) == 38 and len(result.groups) == 2
    assert sum(i.is_f5a for i in result.items) == 36
    assert result.count("EXISTING") == 1 and result.count("NEW") == 33
    assert len(result.eligible_items) == 32 and result.import_caption == "Importuj 32 zgodne sesje"
    assert "Pominięto 2 sesje pilotażowe należące do innej serii." in result.summary()
    assert store.file_path.read_bytes() == before
    store.import_mobile_session_plan(result)
    assert result.count("IMPORTED") == 32 and len(store.reports) == 33
    assert store.reports[0].raw["human_review"] == {"approved": True}
    assert store.reports[0].source_path == original.source_path
    assert store.reports[0].source_archive_sha256 == original.source_archive_sha256
    # A separate explicit action can select the pilot group; it cannot relabel it F5A.
    result.select_group((core.F5A_IMPORT_SERIES + "-PILOT", core.F5A_IMPORT_SCENARIO))
    assert len(result.eligible_items) == 1
    store.import_mobile_session_plan(result)
    assert len(store.reports) == 34 and result.count("CONFLICT") == 4
    reloaded = core.MobilePackageExperimentStore(store.root_dir)
    again = plan(tmp_path / "source", reloaded)
    assert len(again.eligible_items) == 0 and again.count("EXISTING") == 34


def test_UX02_T09_T10_unknown_groups_require_explicit_choice(tmp_path):
    archive(tmp_path / "source" / "a.alprsession", series="A", sid="unknown-a")
    archive(tmp_path / "source" / "b.alprsession", series="B", sid="unknown-b")
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    result = plan(tmp_path / "source")
    assert result.selected_group is None and not result.eligible_items
    with pytest.raises(ValueError, match="Wybierz jawnie"):
        store.import_mobile_session_plan(result)
    assert not store.file_path.exists()
    result.select_group(("B", core.F5A_IMPORT_SCENARIO))
    store.import_mobile_session_plan(result)
    assert len(store.reports) == 1 and core._stored_session_id(store.reports[0]) == "unknown-b"


def test_UX02_T08_file_route_also_cannot_bypass_f5a_policy(tmp_path, settings, monkeypatch):
    def wrong(data): data["session.json"]["execution"]["roi_budget_policy"] = "r2_two_roi"
    path = archive(tmp_path / "wrong.alprsession", variant="R1", edit=wrong)
    store = core.MobilePackageExperimentStore(tmp_path / "store")
    with pytest.raises(ValueError) as exc:
        store.import_mobile_report_file(path)
    assert exc.value.codes == ("VARIANT_POLICY_CONFLICT",) and not store.file_path.exists()
    browser = browser_for_import(tmp_path, monkeypatch, [path])
    browser.import_report()
    assert not browser.store.file_path.exists() and not browser.store.reports


def test_UX02_T21_T22_mt_counts_never_mix_populations(tmp_path, monkeypatch):
    paths = ux02_tree(tmp_path)
    original = core.read_mobile_report_bundles
    def read(path):
        bundle = original(path)[0]
        counts = (45, 12) if path == paths[0] else (3, 1) if path == paths[-1] else (0, 0)
        roles = [dict(postprocess_status="NOT_REACHED", cancel_reason=reason)
                 for reason, n in zip(("scene_superseded", "session_stopped"), counts) for _ in range(n)]
        return (replace(bundle, scene_evidence={"roles": roles}),)
    monkeypatch.setattr(core, "read_mobile_report_bundles", read)
    result = plan(tmp_path)
    assert result.mt_counts(result.items) == (24, {"scene_superseded": 48, "session_stopped": 13})
    assert result.mt_counts(i for i in result.items if i.is_f5a) == (23, {"scene_superseded": 45, "session_stopped": 12})
    assert result.mt_counts(i for i in result.items if not i.is_f5a) == (1, {"scene_superseded": 3, "session_stopped": 1})
    assert len(result.eligible_items) == 32
    text = result.summary()
    assert "Dowody MT — Cały skan:" in text and "Dowody MT — Właściwa seria F5A:" in text and "Dowody MT — Inne serie:" in text


def test_UX02_T01_T11_T12_picker_stays_same_window_no_profile(tmp_path, settings, monkeypatch):
    import tkinter as tk
    source = tmp_path / "source"; archive(source / "one.alprsession")
    chosen = []
    monkeypatch.setattr(gui.filedialog, "askdirectory", lambda **kw: chosen.append(kw) or str(source))
    def check(dialog):
        assert not any(w.winfo_class() == "TCombobox" for w in widgets(dialog))
        window_id = dialog.winfo_id()
        buttons = {w.cget("text"): w for w in widgets(dialog) if w.winfo_class() == "TButton" and w.winfo_ismapped()}
        buttons["Wybierz folder…"].invoke()
        buttons["Sprawdź folder z podfolderami"].invoke()
        flow = dialog._mobile_import_flow
        assert flow.phase == "scanning" and dialog.winfo_exists() and dialog.winfo_id() == window_id
        assert "Sprawdzanie archiwów" == flow.title.get()
        pump(dialog, lambda: flow.phase == "result")
        assert dialog.winfo_id() == window_id and flow.title.get() == "Wynik sprawdzenia"
        assert len([w for w in dialog.master.winfo_children() if isinstance(w, tk.Toplevel)]) == 1
        assert flow.plan.root == source and flow.import_button.cget("text") == "Importuj 1 zgodną sesję"
    assert run_picker(source, check) == ()
    assert chosen[0]["mustexist"]


def test_UX02_T13_T25_worker_progress_only_queue_main_thread_tk(tmp_path, flow_gui, monkeypatch):
    import tkinter as tk
    root, browser, dialog, flow = flow_gui
    source = tmp_path / "source"; archive(source / "one.alprsession")
    main = threading.get_ident(); called = []; release = threading.Event()
    native = gui.plan_mobile_session_import
    def scan(*a, progress, **kw):
        called.append(threading.get_ident())
        progress(1, 3, source / "current.alprsession")
        assert release.wait(8)
        return native(*a, progress=progress, **kw)
    original_after = tk.Misc.after
    def after(widget, *a, **kw):
        assert threading.get_ident() == main
        return original_after(widget, *a, **kw)
    monkeypatch.setattr(gui, "plan_mobile_session_import", scan)
    monkeypatch.setattr(tk.Misc, "after", after)
    flow.scan(source)
    try:
        pump(root, lambda: flow.progress_text.get() == "Sprawdzono 1 z 3 archiwów")
        assert flow.current_file.get() == "Sprawdzanie: current.alprsession"
        assert flow.progress.cget("mode") == "determinate" or str(flow.progress.cget("mode")) == "determinate"
        assert dialog.winfo_exists() and not browser.store.file_path.exists()
    finally:
        release.set()
    pump(root, lambda: flow.phase == "result")
    assert called and called[0] != main


def test_UX02_T14_T27_cancel_scan_then_rescan_in_same_dialog(tmp_path, flow_gui, monkeypatch):
    root, browser, dialog, flow = flow_gui
    source = tmp_path / "source"; archive(source / "one.alprsession")
    native = gui.plan_mobile_session_import
    def scan(*a, cancelled, **kw):
        while not cancelled(): time.sleep(.005)
        return native(*a, cancelled=cancelled, **kw)
    monkeypatch.setattr(gui, "plan_mobile_session_import", scan)
    flow.scan(source); flow.cancel_button.invoke()
    pump(root, lambda: flow.phase == "result")
    assert flow.plan.cancelled and dialog.winfo_exists() and not browser.store.file_path.exists()
    assert str(flow.import_button.cget("state")) == "disabled"
    flow.back_button.invoke()
    assert flow.phase == "selection"
    monkeypatch.setattr(gui, "plan_mobile_session_import", native)
    flow.scan(source); pump(root, lambda: flow.phase == "result")
    assert not flow.plan.cancelled and len(flow.plan.eligible_items) == 1


def test_UX02_T15_T16_T17_T28_copy_confirm_real_counts(tmp_path, flow_gui):
    root, browser, dialog, flow = flow_gui
    source = tmp_path / "source"; archive(source / "one.alprsession")
    flow.scan(source); pump(root, lambda: flow.phase == "result")
    assert not browser.store.file_path.exists()
    flow.copy_button.invoke()
    assert dialog.clipboard_get() == flow.plan.summary()
    assert flow.stats_vars[-1].get() == "Gotowe do importu: 1"
    assert not flow.group_box.winfo_ismapped()
    flow.import_button.invoke()
    assert flow.phase == "importing" and dialog.winfo_exists()
    pump(root, lambda: flow.phase == "finished")
    assert len(browser.store.reports) == 1 and "Zaimportowano: 1" in flow.note.get()
    assert "Już istniały: 0" in flow.note.get() and "Nie zaimportowano: 0" in flow.note.get()
    prefs = gui.mobile_import_preferences
    assert prefs.SESSION.get(prefs.SECTION, prefs.FOLDER) == str(source)
    flow.copy_button.invoke()
    assert dialog.clipboard_get() == flow.plan.summary(completed=True)


@pytest.mark.parametrize("close_request", [False, True])
def test_UX02_T18_partial_write_cancel_keeps_counts_and_window(tmp_path, flow_gui, monkeypatch, close_request):
    root, browser, dialog, flow = flow_gui
    source = tmp_path / "source"
    for block in "ABC": archive(source / f"{block}.alprsession", block=block)
    flow.scan(source); pump(root, lambda: flow.phase == "result")
    original = browser.store.import_mobile_session_plan
    wrote = threading.Event()
    def commit(plan, *, progress, cancelled):
        def update(*a):
            progress(*a); wrote.set()
            while not cancelled(): time.sleep(.005)
        return original(plan, progress=update, cancelled=cancelled)
    monkeypatch.setattr(browser.store, "import_mobile_session_plan", commit)
    flow.commit(); pump(root, wrote.is_set)
    if close_request:
        flow.close_callback()  # X/Esc requests cancellation while writing; retain the outcome.
    else:
        flow.cancel_button.invoke()
    pump(root, lambda: flow.phase == "finished")
    assert dialog.winfo_exists() and flow.plan.cancelled
    assert len(browser.store.reports) == 1 and "Zaimportowano: 1" in flow.note.get() and "Nie zaimportowano: 2" in flow.note.get()
    assert len(core.MobilePackageExperimentStore(browser.store.root_dir).reports) == 1


@pytest.mark.parametrize("phase", ["selection", "scanning", "result"])
def test_UX02_T26_close_without_write_stops_worker_safely(tmp_path, flow_gui, monkeypatch, phase):
    root, browser, dialog, flow = flow_gui
    source = tmp_path / "source"; archive(source / "one.alprsession")
    if phase == "scanning":
        native = gui.plan_mobile_session_import
        def slow(*a, cancelled, **kw):
            while not cancelled(): time.sleep(.005)
            return native(*a, cancelled=cancelled, **kw)
        monkeypatch.setattr(gui, "plan_mobile_session_import", slow)
    if phase != "selection":
        flow.scan(source)
        if phase == "result": pump(root, lambda: flow.phase == "result")
    flow.close_callback()
    assert flow.closed and flow.cancel_event.is_set() and not browser.store.file_path.exists()


def test_UX02_T10_group_action_changes_cta_without_changing_source(tmp_path, flow_gui):
    root, browser, dialog, flow = flow_gui
    paths = ux02_tree(tmp_path / "source")
    browser.store.import_mobile_report_file(paths[-2])
    flow.scan(tmp_path / "source"); pump(root, lambda: flow.phase == "result")
    assert flow.import_button.cget("text") == "Importuj 32 zgodne sesje"
    assert len(flow.files.get_children("good")) == 32
    assert len(flow.files.get_children("bad")) == 4
    assert len(flow.files.get_children("other")) == 2
    assert len(flow.files.get_children("existing")) == 1
    pilot = next(k for k, group in flow.group_keys.items() if group != core.F5A_IMPORT_GROUP)
    flow.group_tree.selection_set(pilot)
    assert flow.import_button.cget("text") == "Importuj 32 zgodne sesje"  # selection alone is not authorization
    flow.group_button.invoke()
    assert flow.import_button.cget("text") == "Importuj 1 zgodną sesję"
    assert sum(i.status == "CONFLICT" for i in flow.plan.items) == 4
    assert len(browser.store.reports) == 1


@pytest.mark.parametrize("action", ["escape", "x"])
@pytest.mark.parametrize("phase", ["selection", "scanning", "result"])
def test_UX02_T26_real_picker_bindings_close_safely(tmp_path, settings, action, phase):
    source = tmp_path / "source"; archive(source / "one.alprsession")
    def close(dialog):
        flow = dialog._mobile_import_flow
        if phase != "selection":
            next(w for w in widgets(dialog) if w.winfo_class() == "TButton"
                 and w.cget("text") == "Sprawdź folder z podfolderami").invoke()
            if phase == "result": pump(dialog, lambda: flow.phase == "result")
        dialog.focus_force(); dialog.update()
        if action == "escape": dialog.event_generate("<Escape>")
        else: dialog.tk.call(dialog.protocol("WM_DELETE_WINDOW"))
        assert flow.closed and flow.cancel_event.is_set() and not dialog.winfo_exists()
    assert run_picker(source, close) == ()
    assert not (source.parent / "private-picker-store" / core.MobilePackageExperimentStore.FILE_NAME).exists()


def test_UX02_result_actions_remain_visible_at_minimum_size(tmp_path, flow_gui):
    root, browser, dialog, flow = flow_gui
    ux02_tree(tmp_path / "source")
    dialog.geometry("780x650")
    flow.scan(tmp_path / "source"); pump(root, lambda: flow.phase == "result")
    for button in (flow.back_button, flow.copy_button, flow.cancel_button, flow.import_button):
        assert button.winfo_ismapped()
        assert button.winfo_rootx() >= dialog.winfo_rootx()
        assert button.winfo_rootx() + button.winfo_width() <= dialog.winfo_rootx() + dialog.winfo_width()
        assert button.winfo_rooty() + button.winfo_height() <= dialog.winfo_rooty() + dialog.winfo_height()
    assert all(v.get() in flow.plan.summary() for v in flow.stats_vars)
