"""AKT83: per-session, revisioned, no-overwrite export of Mobile review reports.

These are filesystem-only tests on temporary directories. They do not load a
phone archive, modify GT, alter model weights, or change the experiment tree.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]
HELPER = HERE / "auto_annotation_tool" / "gui" / "mobile_review_export_layout.py"
HOST = HERE / "auto_annotation_tool" / "gui" / "z4_mobile_sample_review.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("mobile_review_export_layout_akt83", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def root(tmp_path):
    result = tmp_path / "E-MOB-QINT8-01" / "03_wyniki"
    result.mkdir(parents=True)
    (result / "porownanie").mkdir()
    (result / "review.json").write_text("HISTORYCZNY_EKSPORT", encoding="utf-8")
    return result


def fake_session(sid="exp-1792701720679-5318a7d2", rev=101):
    review = SimpleNamespace(
        review_id="review-test-id", review_revision=rev, review_status="COMPLETED",
        review_mode="assisted", source_archive_sha256="a" * 64,
        provenance={"variant_id": "DAY_F32", "experiment_session": {"scenario_id": "PAIR_DAY_01"}})
    class Session:
        session_id = sid
        path = Path("alpr_session_20261022_123456.alprsession")
        def __init__(self):
            self.review = review
            self.verify_count = 0
        def verify_source(self, force=False):
            self.verify_count += 1
        def statistics(self):
            return {"summary": {"canvas_exact": 15}}
        def export(self, path):
            paths = []
            for name in ("review.json", "summary.csv", "subjects.csv", "character_confusion.csv", "canvas_review_plates.csv"):
                p = path/name
                p.write_text(f"{self.session_id}  rev:{self.review.review_revision}\n",encoding="utf-8")
                paths.append(p)
            return paths
    return Session()


def test_directory_preview_does_not_create_anything(tmp_path):
    m = load_helper()
    base = root(tmp_path)
    p = m.next_export_path(base, "exp-123", 5)
    assert p == base/"exp-123"/"review_r000005_export_001"
    assert not p.parent.exists()
    assert not p.exists()


def test_reservation_prevents_overwrite_same_session_revision(tmp_path):
    m = load_helper()
    base = root(tmp_path)
    one = m.reserve_export_path(base, "exp-123", 5)
    (one/"review.json").write_text("old")
    two = m.reserve_export_path(base, "exp-123", 5)
    assert one.name == "review_r000005_export_001"
    assert two.name == "review_r000005_export_002"
    assert (one/"review.json").read_text() == "old"
    assert (base/"review.json").read_text() == "HISTORYCZNY_EKSPORT"
    assert list((base/"porownanie").iterdir()) == []


def test_distinct_sessions_and_revisions(tmp_path):
    m = load_helper()
    base = root(tmp_path)
    a = m.reserve_export_path(base, "exp-aaa", 1)
    b = m.reserve_export_path(base, "exp-bbb", 1)
    c = m.reserve_export_path(base, "exp-aaa", 2)
    assert a.parent != b.parent
    assert a.parent == c.parent
    assert a.name != c.name


@pytest.mark.parametrize("sid", ["../foo", "../porownanie", "a/b", "foo\\bar", "", ".", "..", "a"*120])
def test_bad_session_id_rejected(tmp_path,sid):
    m = load_helper()
    base = root(tmp_path)
    with pytest.raises(ValueError):
        m.reserve_export_path(base, sid, 1)
    assert not (tmp_path/"foo").exists()


def test_wrong_target_and_bad_revision_rejected_without_writes(tmp_path):
    m = load_helper()
    base = root(tmp_path)
    with pytest.raises(ValueError):
        m.reserve_export_path(base/"porownanie", "exp-123", 1)
    with pytest.raises(ValueError):
        m.reserve_export_path(base, "exp-123", -1)
    assert not (base/"exp-123").exists()


def test_manifest_contains_sha_for_files_and_thesis_directory(tmp_path):
    m = load_helper()
    base = root(tmp_path)
    session = fake_session()
    target = m.reserve_export_path(base, session.session_id, session.review.review_revision)
    files = session.export(target)
    for name in ("mt_pose_geometry.csv", "mt_multi_plate_verified.json"):
        file = target/name
        file.write_text("test",encoding="utf-8")
        files.append(file)
    thesis = target/"thesis_foo"
    thesis.mkdir()
    (thesis/"thesis_metrics.csv").write_text("abc",encoding="utf-8")
    (thesis/"sources").mkdir()
    (thesis/"sources"/"summary.csv").write_text("def",encoding="utf-8")
    files += [thesis/"sources", thesis/"thesis_metrics.csv"]
    manifest = m.finalize_export_manifest(session, target, files)
    info = json.loads(manifest.read_text(encoding="utf-8"))
    assert info["session_id"] == session.session_id
    assert info["review_revision"] == 101
    assert info["review_status"] == "COMPLETED"
    assert info["source_archive_sha256"] == "a"*64
    assert info["variant_as_recorded"] == "DAY_F32"
    assert "r01" not in json.dumps(info)
    assert {x["path"] for x in info["files"]} >= {"review.json", "thesis_foo/thesis_metrics.csv", "thesis_foo/sources/summary.csv"}
    for entry in info["files"]:
        p = target/entry["path"]
        assert entry["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        assert entry["bytes"] == p.stat().st_size
    assert session.verify_count == 1


def test_export_failure_marked_in_new_directory_only(tmp_path):
    m=load_helper()
    base=root(tmp_path)
    target=m.reserve_export_path(base, "exp-123", 1)
    m.mark_export_incomplete(target, "Przerwany zapis")
    assert "Przerwany zapis" in (target/"EKSPORT_NIEKOMPLETNY.txt").read_text()
    assert not (target/"export_manifest.json").exists()
    assert (base/"review.json").read_text()=="HISTORYCZNY_EKSPORT"


def _host_export_function():
    """Run the ACTUAL exported method in isolation with fake GUI collaborators."""
    tree = ast.parse(HOST.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MobileSampleReviewWindow")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "export")
    fn.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),str(HOST),"exec")
    return code


def test_real_export_callback_creates_separate_session_folders(tmp_path):
    m=load_helper()
    base=root(tmp_path)
    message_log=[]
    statuses=[]

    def fake_thesis(session, statistics, rows, export_files, target):
        (target/"thesis_generated").mkdir()
        (target/"thesis_generated"/"fig1.png").write_bytes(b"abc")
        (target/"thesis_generated"/"sources").mkdir()
        (target/"thesis_generated"/"sources"/"summary.csv").write_bytes(b"cba")
        return [target/"thesis_generated"/"fig1.png", target/"thesis_generated"/"sources"]

    def fake_scientific(summary, target):
        path=target/"ocena_naukowa.md"
        path.write_text("metodyka",encoding="utf-8")
        return [path]

    globals_ = {
        'filedialog': SimpleNamespace(askdirectory=lambda **k: str(base)),
        'messagebox': SimpleNamespace(askyesno=lambda *a,**k: (message_log.append((a,k)) or True),
                                      showerror=lambda *a,**k: (_ for _ in ()).throw(AssertionError("bad dir"))),
        'Path': Path, 'csv': csv, 'os': os, 'json': json,
        'validate_export_destination': m.validate_export_destination,
        'next_export_path': m.next_export_path,
        'reserve_export_path': m.reserve_export_path,
        'finalize_export_manifest': m.finalize_export_manifest,
        'mark_export_incomplete': m.mark_export_incomplete,
        '_pose_geometry_rows': lambda session: [{"session_id": session.session_id, "source_archive_sha256": "a"*64, "attempt_id": "mt1", "metric_status":"measured"}],
        'export_thesis_bundle': fake_thesis,
        'make_scientific_summary':lambda *args: {},
        'write_scientific_companion':fake_scientific,
        'multi_scientific_summary':lambda session: {"session_id": session.session_id},
    }
    exec(_host_export_function(),globals_)
    def run(sid,revision):
        session=fake_session(sid,revision)
        class Window:
            pass
        host=Window()
        host.session=session
        host._saving=False
        host._canvas_editor=None
        host.window=object()
        host.status_var=SimpleNamespace(set=lambda s: statuses.append(s))
        def _change(op,*,after):
            op()
            after()
        host._change=_change
        globals_['export'](host)
        return statuses[-1]

    assert "exp-001" in run("exp-001", 7)
    assert "exp-001" in run("exp-001", 7)
    assert "exp-002" in run("exp-002", 7)
    assert (base/"exp-001"/"review_r000007_export_001"/"export_manifest.json").is_file()
    assert (base/"exp-001"/"review_r000007_export_002"/"export_manifest.json").is_file()
    assert (base/"exp-002"/"review_r000007_export_001"/"export_manifest.json").is_file()
    assert (base/"review.json").read_text()=="HISTORYCZNY_EKSPORT"
    assert (base/"porownanie").is_dir()
    assert len(message_log)==3
