# Z4MOBREV009_THESIS_EXPORT_TESTS_20261017
"""No production archive is required: controlled synthetic session and metric values."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_annotation_tool.ranking.mobile_thesis_report import (
    SCHEMA, _histogram, _metric_rows, _tex_escape, export_thesis_bundle,
)


@pytest.fixture
def sample(tmp_path):
    archive = tmp_path / "session.alprsession"
    archive.write_bytes(b"synthetic original session: stable and not a real user archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    review = SimpleNamespace(source_archive_sha256=digest, review_revision=12, review_status="IN_PROGRESS",
                             review_mode="assisted", provenance={"MT": "sample-mt", "MZ": "sample-mz"})
    bundle = SimpleNamespace(experiment_session={"series_id": "F4-TV-FP32-A12-20261016", "variant": "F4-TV-FP32-B",
                                                  "scenario_id": "TV6X10_STATIC_B", "replicate_index": 1},
                             metadata={})
    session = SimpleNamespace(session_id="exp-test_12", path=archive, review=review, bundle=bundle)
    stats = {"summary": {"subject_count": 6, "reviewed_subjects": 3, "not_reviewed_subjects": 3,
                         "evaluable_subjects": 4, "subjects_with_exact_read": 2, "subject_success_rate": .5,
                         "evaluable_reads": 4, "exact_reads": 2, "incorrect_reads": 1, "no_reads": 1,
                         "exact_read_rate": .5, "no_read_rate": .25, "gt_characters": 28, "cer": .25,
                         "mt_invocation_count": 7, "evaluable_mt_invocations": 4,
                         "mt_successful_invocations": 3, "mt_localization_success_rate": .75,
                         "crop_count": 5, "mt_false_detections": 1}}
    rows = [
        {"session_id": session.session_id, "source_archive_sha256": digest, "attempt_id": "a1", "eligible": True,
         "metric_status": "measured", "nmce_pct": 1.6, "polygon_iou_pct": 82.4},
        {"session_id": session.session_id, "source_archive_sha256": digest, "attempt_id": "a2", "eligible": True,
         "metric_status": "missing_manual_gt", "nmce_pct": None, "polygon_iou_pct": None},
        {"session_id": session.session_id, "source_archive_sha256": digest, "attempt_id": "a3", "eligible": False,
         "metric_status": "not_confirmed_plate", "nmce_pct": None, "polygon_iou_pct": None},
    ]
    csv_path = tmp_path / "mt_pose_geometry.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return session, stats, rows, csv_path


def _report(sample, tmp_path):
    session, stats, rows, csv_path = sample
    outputs = export_thesis_bundle(session, stats, rows, [csv_path], tmp_path)
    return outputs[0].parent


def test_report_has_auditable_artifacts_and_no_mutation(sample, tmp_path):
    session, stats, rows, csv_path = sample
    before_file = csv_path.read_bytes()
    before_archive = session.path.read_bytes()
    before_rows = [dict(row) for row in rows]
    folder = _report(sample, tmp_path)
    assert (folder / "thesis_report.md").is_file()
    assert (folder / "thesis_metrics.csv").is_file()
    assert (folder / "thesis_manifest.json").is_file()
    assert (folder / "thesis_figures.tex").is_file()
    assert (folder / "thesis_tables.tex").is_file()
    assert csv_path.read_bytes() == before_file
    assert session.path.read_bytes() == before_archive
    assert rows == before_rows


def test_manifest_binds_session_sha_provenance_revision_and_scenario(sample, tmp_path):
    session, stats, rows, csv_path = sample
    folder = _report(sample, tmp_path)
    m = json.loads((folder / "thesis_manifest.json").read_text(encoding="utf-8"))
    assert m["schema"] == SCHEMA
    assert m["session"]["source_archive_sha256"] == session.review.source_archive_sha256
    assert m["session"]["source_verified_at_export"] is True
    assert m["session"]["scenario_id"] == "TV6X10_STATIC_B"
    assert m["model_provenance"] == {"MT": "sample-mt", "MZ": "sample-mz"}
    assert m["review"]["review_revision"] == 12
    assert m["review"]["quality_status"] == "PRELIMINARY_INCOMPLETE_REVIEW"
    assert m["inputs"][0]["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert all((folder / item["file"]).is_file() and hashlib.sha256((folder / item["file"]).read_bytes()).hexdigest() == item["sha256"]
               for item in m["outputs"])


def test_distribution_only_counts_genuine_manual_gt(sample, tmp_path):
    folder = _report(sample, tmp_path)
    m = json.loads((folder / "thesis_manifest.json").read_text(encoding="utf-8"))
    assert m["pose_status_counts"] == {"measured": 1, "missing_manual_gt": 1, "not_confirmed_plate": 1}
    assert (folder / "fig_pose_iou.png").is_file()
    assert (folder / "fig_pose_nmce.png").is_file()
    with (folder / "thesis_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        metrics = {r["metric"]: r for r in csv.DictReader(f)}
    assert metrics["pose_measured"]["value"] == "1"
    assert metrics["pose_pending"]["value"] == "1"
    assert metrics["pose_iou_mean_pct"]["value"] == "82.4"
    assert metrics["pose_nmce_mean_pct"]["value"] == "1.6"


def test_missing_gt_must_not_render_pose_plots_or_fake_percent(sample, tmp_path):
    session, stats, rows, csv_path = sample
    rows[0]["metric_status"] = "missing_manual_gt"
    rows[0]["nmce_pct"] = None
    rows[0]["polygon_iou_pct"] = None
    folder = _report(sample, tmp_path)
    assert not (folder / "fig_pose_iou.png").exists()
    assert not (folder / "fig_pose_nmce.png").exists()
    with (folder / "thesis_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        d = {r["metric"]: r for r in csv.DictReader(f)}
    assert d["pose_measured"]["value"] == "0"
    assert d["pose_iou_mean_pct"]["status"] == "unavailable"
    assert d["pose_iou_mean_pct"]["value"] == ""


def test_rates_require_positive_denominator_even_when_rate_is_zero(sample):
    _, stats, rows, _ = sample
    stats["summary"].update(evaluable_reads=0, exact_reads=0, exact_read_rate=0, no_reads=0,
                            no_read_rate=0, gt_characters=0, cer=0)
    metrics, _, _ = _metric_rows(stats["summary"], rows)
    d = {m["metric"]: m for m in metrics}
    assert all(d[key]["status"] == "unavailable" and d[key]["value"] is None
               for key in ("exact_read_rate", "no_read_rate", "cer"))


def test_inconsistent_ratio_is_blocked_not_published(sample):
    _, stats, rows, _ = sample
    stats["summary"]["exact_reads"] = 3
    metrics, warnings, _ = _metric_rows(stats["summary"], rows)
    assert {m["metric"]: m for m in metrics}["exact_read_rate"]["value"] is None
    assert any("exact_read_rate" in w for w in warnings)


def test_completed_review_is_named_completed_only_with_full_gt(sample, tmp_path):
    session, stats, rows, _ = sample
    session.review.review_status = "COMPLETED"
    stats["summary"]["not_reviewed_subjects"] = 0
    rows = [row for row in rows if row["attempt_id"] != "a2"]
    folder = export_thesis_bundle(session, stats, rows, [sample[3]], tmp_path)[0].parent
    m = json.loads((folder / "thesis_manifest.json").read_text(encoding="utf-8"))
    assert m["review"]["quality_status"] == "REVIEW_COMPLETED"


def test_sha_mismatch_blocks_without_output(sample, tmp_path):
    session, stats, rows, csv_path = sample
    session.path.write_bytes(b"the original file was changed")
    with pytest.raises(ValueError, match="SHA-256 archiwum"):
        export_thesis_bundle(session, stats, rows, [csv_path], tmp_path)
    assert not list(tmp_path.glob("thesis_*"))


def test_mixed_source_rows_are_rejected(sample, tmp_path):
    session, stats, rows, csv_path = sample
    rows[0]["source_archive_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Mieszane źródła"):
        export_thesis_bundle(session, stats, rows, [csv_path], tmp_path)


def test_missing_export_file_is_rejected(sample, tmp_path):
    session, stats, rows, csv_path = sample
    csv_path.unlink()
    with pytest.raises(ValueError, match="nie istnieje"):
        export_thesis_bundle(session, stats, rows, [csv_path], tmp_path)


def test_source_outside_export_directory_is_rejected(sample, tmp_path):
    session, stats, rows, csv_path = sample
    other = tmp_path.parent / "unauthorized_input.csv"
    other.write_bytes(b"test")
    try:
        with pytest.raises(ValueError, match="poza katalogiem"):
            export_thesis_bundle(session, stats, rows, [other], tmp_path)
    finally:
        other.unlink(missing_ok=True)


def test_fresh_versions_never_overwrite_previous_exports(sample, tmp_path):
    dir1 = _report(sample, tmp_path)
    content1 = (dir1 / "thesis_report.md").read_bytes()
    dir2 = _report(sample, tmp_path)
    assert dir1 != dir2 and dir1.is_dir() and dir2.is_dir()
    assert (dir1 / "thesis_report.md").read_bytes() == content1


def test_latex_labels_are_fixed_and_paths_are_local(sample, tmp_path):
    folder = _report(sample, tmp_path)
    figures = (folder / "thesis_figures.tex").read_text(encoding="utf-8")
    tables = (folder / "thesis_tables.tex").read_text(encoding="utf-8")
    assert "\\label{fig:alpr_mobile_pose_iou_" in figures
    assert f"{{{folder.name}/fig_pose_iou.png}}" in figures
    assert "\\label{tab:alpr_mobile_review_" in tables
    assert "\\begin{table}" in tables
    assert "\\begin{figure}" in figures


def test_escape_tex_and_histogram_boundaries():
    assert _tex_escape("x_1% &") == r"x\_1\% \&"
    assert _histogram([0, 20, 99.9, 100], [0, 20, 40, 60, 80, 100]) == [1, 1, 0, 0, 2]
    with pytest.raises(ValueError):
        _histogram([-1], [0, 1])


def test_empty_statistics_marks_all_ratios_unavailable(sample, tmp_path):
    session, _, rows, csv_path = sample
    folder = export_thesis_bundle(session, {"summary": {}}, rows, [csv_path], tmp_path)[0].parent
    assert not (folder / "fig_quality_rates.png").exists()
    text = (folder / "thesis_report.md").read_text(encoding="utf-8")
    assert "brak danych" in text


def test_sanitized_session_folder_does_not_traverse(sample, tmp_path):
    session, stats, rows, csv_path = sample
    session.session_id = "..\\../../***"
    folder = export_thesis_bundle(session, stats, rows, [csv_path], tmp_path)[0].parent
    assert folder.parent == tmp_path.resolve()
    assert ".." not in folder.name


def test_gui_export_remains_one_button_and_invokes_thesis_backend():
    import ast
    gui_path = Path(__file__).resolve().parents[1] / "auto_annotation_tool" / "gui" / "z4_mobile_sample_review.py"
    tree = ast.parse(gui_path.read_text(encoding="utf-8"))
    ui_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MobileSampleReviewWindow")
    method = next(n for n in ui_cls.body if isinstance(n, ast.FunctionDef) and n.name == "export")
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == "export" for n in calls)
    assert any(isinstance(n.func, ast.Name) and n.func.id == "export_thesis_bundle" for n in calls)
    assert any(isinstance(n.func, ast.Name) and n.func.id == "_pose_geometry_rows" for n in calls)


def test_generated_charts_are_valid_png_images(sample, tmp_path):
    from PIL import Image
    folder = _report(sample, tmp_path)
    for image_path in folder.glob("fig_*.png"):
        with Image.open(image_path) as im:
            assert im.format == "PNG"
            assert im.width >= 1400 and im.height >= 800
            im.verify()


def test_actual_gui_export_method_writes_thesis_bundle(sample, tmp_path):
    """Execute GUI export() itself without opening Tk or importing unrelated runtime components."""
    import ast
    import os
    from auto_annotation_tool.ranking.mobile_thesis_report import export_thesis_bundle

    gui = Path(__file__).resolve().parents[1] / "auto_annotation_tool" / "gui" / "z4_mobile_sample_review.py"
    tree = ast.parse(gui.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MobileSampleReviewWindow")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "export")
    exec_body = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(exec_body)
    session, stats, rows, csv_path = sample
    session.export = lambda dest: [csv_path]
    session.statistics = lambda: stats

    class State:
        message = None
        def set(self, message):
            self.message = message

    ns = {"csv": csv, "os": os, "Path": Path,
          "_pose_geometry_rows": lambda s: rows,
          "export_thesis_bundle": export_thesis_bundle,
          "filedialog": SimpleNamespace(askdirectory=lambda **kw: str(tmp_path))}
    exec(compile(exec_body, str(gui), "exec"), ns)
    window = SimpleNamespace(window=object(), session=session, status_var=State(),
                             _change=lambda fn, after: (fn(), after()))
    ns["export"](window)
    dirs = list(tmp_path.glob("thesis_*"))
    assert len(dirs) == 1
    assert (dirs[0] / "thesis_manifest.json").is_file()
    assert "raport do pracy" in window.status_var.message


def test_snapshot_is_immutable_when_parent_csv_overwritten(sample, tmp_path):
    session, stats, rows, csv_path = sample
    first_data = csv_path.read_bytes()
    first_folder = _report(sample, tmp_path)
    first_manifest = json.loads((first_folder / "thesis_manifest.json").read_text(encoding="utf-8"))
    snapshot = first_folder / first_manifest["inputs"][0]["snapshot"]
    assert snapshot.read_bytes() == first_data
    csv_path.write_text("csv,overwritten\n", encoding="utf-8")
    assert snapshot.read_bytes() == first_data
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == first_manifest["inputs"][0]["sha256"]
    assert any(entry["file"] == first_manifest["inputs"][0]["snapshot"] for entry in first_manifest["outputs"])
