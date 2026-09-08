import copy
import csv
import io
import json
from pathlib import Path
import zipfile
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from auto_annotation_tool.ranking.mobile_human_review import (
    MobileReviewSession, align_plate_text, normalize_registration,
)
from auto_annotation_tool.ranking.mobile_package_experiments import (
    MobilePackageExperimentStore, _file_sha256, read_mobile_report_bundle,
    iter_full_attempt_rows, read_mobile_sample_image, score_mobile_report,
)


def make_bundle(path, *, samples=None, attempts=None, annotations=None, image_size=(280, 80), image_text="", state="COMPLETED", evidence_images=None):
    samples = samples if samples is not None else [dict(capture_id="c1", subject_key="s/sg-1/entity-1", prediction="WI1234A", attempt_id="a1")]
    payload = {"schema": "alpr.mobile_benchmark_report.v1", "report_id": "session-s", "session_id": "s",
               "package_id": "pkg", "variant_id": "ncnn-fp16", "state": state,
               "latency": {"pipeline_p95_ms": 120}, "memory": {"ram_peak_mb": 128},
               "quality": {"exact_match_rate": .2, "cer": .4},
               "model_provenance": {"plate": {"model_id": "mt", "variant_id": "fp16", "checkpoint_sha256": "abc"}}}
    def csv_text(rows):
        columns = list(dict.fromkeys(key for row in rows for key in row)) or ["attempt_id"]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()
    image = io.BytesIO()
    plate = Image.new("RGB", image_size, "white")
    if image_text:
        draw = ImageDraw.Draw(plate)
        draw.rectangle((2, 2, image_size[0] - 3, image_size[1] - 3), outline="black", width=3)
        try:
            font = ImageFont.truetype("arialbd.ttf", 36)
        except OSError:
            font = ImageFont.load_default()
        draw.text((22, 17), image_text, font=font, fill="black")
    plate.save(image, format="JPEG")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", json.dumps(payload))
        archive.writestr("samples/index.csv", csv_text(samples))
        archive.writestr("samples/annotations.jsonl", "".join(json.dumps(row) + "\n" for row in (annotations or [])))
        for row in samples:
            archive.writestr(f"samples/crops/{row['capture_id']}.jpg", image.getvalue())
        if attempts is not None:
            archive.writestr("samples/schema.json", json.dumps({"schema": "alpr.mobile_research_samples.v2", "human_review": "desktop"}))
            archive.writestr("samples/attempts.csv", csv_text(attempts))
            for row in attempts:
                if row.get("evidence_entry"):
                    archive.writestr(row["evidence_entry"], (evidence_images or {}).get(row["evidence_entry"], image.getvalue()))
    return path


def attempt(key="a1", *, subject="s/sg-1/entity-1", status="VALID_QUAD", prediction="WI1234A", **extra):
    return dict(attempt_id=key, subject_key=subject, session_id="s", mt_status=status,
                mz_status="READ" if prediction else "NO_CHARACTERS", prediction=prediction,
                evidence_entry=f"samples/evidence/{key}.jpg", attempt_started_elapsed_nanos=1_000_000_000,
                capture_source="normal", **extra)


def load(path, tmp_path):
    return MobileReviewSession.open(path, sidecar_path=tmp_path / "review.json")


@pytest.mark.parametrize("prediction,counts", [
    ("WI1234A", (7, 0, 0, 0)), ("WI12B4A", (6, 1, 0, 0)),
    ("WI123A", (6, 0, 1, 0)), ("WI12344A", (7, 0, 0, 1)), ("", (0, 0, 7, 0)),
])
def test_d1_to_d5_alignment(prediction, counts):
    result = align_plate_text("WI1234A", prediction)
    assert (result.correct_characters, result.incorrect_characters, result.missing_characters, result.extra_characters) == counts
    assert result.cer == pytest.approx(sum(counts[1:]) / 7)
    assert result.exact_match == (prediction == "WI1234A")


def test_normalization_empty_gt_and_repeated_alignment():
    assert normalize_registration(" wi-12 34a ") == "WI1234A"
    assert not align_plate_text("O1", "01").exact_match
    assert align_plate_text("", "ABC").cer is None
    assert align_plate_text("A", "ABCD").cer == 3
    assert align_plate_text("ABAB", "BABA") == align_plate_text("ABAB", "BABA")


def test_new_reader_uses_full_attempts_and_safe_lazy_image(tmp_path):
    attempts = [attempt(f"a{index}") for index in range(1005)]
    path = make_bundle(tmp_path / "new.alprsession", attempts=attempts)
    bundle = read_mobile_report_bundle(path)
    assert bundle.validation.ok
    assert bundle.attempts_available and bundle.attempt_total == 1005
    assert len(bundle.attempt_rows) == 1000
    assert len(list(iter_full_attempt_rows(bundle))) == 1005
    assert bundle.sample_schema["schema"] == "alpr.mobile_research_samples.v2"
    assert read_mobile_sample_image(bundle, "samples/evidence/a1004.jpg").size == (280, 80)
    with pytest.raises(ValueError):
        read_mobile_sample_image(bundle, "samples/crops/../../outside.jpg")
    with pytest.raises(ValueError):
        read_mobile_sample_image(bundle, "samples/crops/c1.jpg", max_entry_bytes=5)
    with pytest.raises(ValueError):
        read_mobile_sample_image(bundle, "samples/crops/c1.jpg", max_dimension=100)


def test_legacy_subject_identity_does_not_use_prediction(tmp_path):
    rows = [dict(capture_id="c1", track_id="7", prediction="ABC"), dict(capture_id="c2", track_id="7", prediction="XYZ"),
            dict(capture_id="c3", track_id="8", prediction="ABC")]
    session = load(make_bundle(tmp_path / "old.alprsession", samples=rows), tmp_path)
    assert not session.attempts_available and session.warnings
    assert len(session.subjects) == 2
    assert all(subject["legacy_identity"] for subject in session.subjects.values())
    key = session.samples["c1"]["subject_key"]
    assert key == session.samples["c2"]["subject_key"] != session.samples["c3"]["subject_key"]
    session.set_subject(key, ground_truth="ABC")
    assert session.statistics()["summary"]["evaluable_reads"] == 2
    assert session.statistics()["summary"]["mt_localization_success_rate"] is None


def test_autosave_reopen_sha_binding_and_original_immutable(tmp_path):
    path = make_bundle(tmp_path / "source.alprsession", attempts=[attempt()])
    source_hash = _file_sha256(path)
    session = load(path, tmp_path)
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="WI1234A")
    assert load(path, tmp_path).review.review_revision == 1
    session.annotate("attempt", "a1", plate_visibility="visible", is_plate=True)
    session.annotate_invocation("a1", visible_plate_count="one")
    session.complete()
    reopened = load(path, tmp_path)
    assert reopened.review.review_status == "COMPLETED"
    assert reopened.review.review_revision == 4
    assert _file_sha256(path) == source_hash
    other = make_bundle(tmp_path / "other.alprsession", samples=[dict(capture_id="other", prediction="XX")])
    with pytest.raises(ValueError, match="SHA-256"):
        load(other, tmp_path)
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="zmieniła"):
        session.set_subject(key, ground_truth="OTHER")


def test_failed_atomic_replace_keeps_saved_and_in_memory_revision(tmp_path):
    session = load(make_bundle(tmp_path / "source.alprsession"), tmp_path)
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="FIRST")
    before = session.sidecar_path.read_bytes()
    with patch("auto_annotation_tool.ranking.mobile_human_review.os.replace", side_effect=OSError("disk error")):
        with pytest.raises(OSError):
            session.set_subject(key, ground_truth="SECOND")
    assert session.sidecar_path.read_bytes() == before
    assert session.review.subjects[key]["ground_truth"] == "FIRST"
    assert not list(tmp_path.glob("*.tmp"))


def test_exact_incorrect_and_no_read_are_distinct(tmp_path):
    samples = [dict(capture_id=f"c{i}", subject_key="s/sg-1/entity-1", prediction=prediction)
               for i, prediction in enumerate(["WI1234A"] * 7 + ["WI12B4A"] * 2 + [""])]
    session = load(make_bundle(tmp_path / "source.alprsession", samples=samples), tmp_path)
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    stats = session.statistics()
    s = stats["summary"]
    assert (s["evaluable_reads"], s["exact_reads"], s["incorrect_reads"], s["no_reads"]) == (10, 7, 2, 1)
    assert s["exact_read_rate"] == .7 and s["no_read_rate"] == .1
    assert (s["gt_characters"], s["incorrect_characters"], s["missing_characters"]) == (70, 2, 7)
    assert s["cer"] == pytest.approx(9 / 70)
    assert stats["character_confusion"] == [dict(ground_truth="3", prediction="B", count=2)]


def test_subject_rate_and_micro_cer(tmp_path):
    samples = [dict(capture_id=f"c{i}", subject_key=f"s/{i}", prediction="ABC" if i < 4 else "") for i in range(5)]
    session = load(make_bundle(tmp_path / "source.alprsession", samples=samples), tmp_path)
    for key in session.subjects:
        session.set_subject(key, ground_truth="ABC")
    assert session.statistics()["summary"]["subject_success_rate"] == .8
    session.set_subject("s/4", ground_truth="ABCDEFGHI")
    assert session.statistics()["summary"]["cer"] == pytest.approx(9 / 21)


def test_mt_miss_invisible_uncertain_cancelled_and_false_detection(tmp_path):
    rows = [attempt("hit"), attempt("miss", status="NO_DETECTION", prediction=""),
            attempt("invalid", status="DETECTION_INVALID_QUAD", prediction=""),
            attempt("no_plate", status="NO_DETECTION", prediction=""), attempt("uncertain"),
            attempt("cancelled", stale_or_cancelled=True), attempt("false")]
    session = load(make_bundle(tmp_path / "source.alprsession", samples=[], attempts=rows), tmp_path)
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="WI1234A")
    for row in rows:
        session.annotate("attempt", row["attempt_id"], plate_visibility="visible", is_plate=True)
        session.annotate_invocation(row["attempt_id"], visible_plate_count="one")
    session.annotate("attempt", "no_plate", plate_visibility="invisible")
    session.annotate("attempt", "uncertain", plate_visibility="uncertain")
    session.annotate_invocation("no_plate", visible_plate_count="none")
    session.annotate_invocation("uncertain", visible_plate_count="uncertain")
    session.annotate("attempt", "false", is_plate=False)
    s = session.statistics()["summary"]
    assert s["evaluable_mt_attempts"] == 4
    assert s["mt_valid_localizations"] == 1
    assert s["mt_no_detections"] == 1 and s["mt_invalid_quads"] == 1
    assert s["mt_false_detections"] == 1
    assert s["mt_localization_success_rate"] == .25
    session.complete()


def test_empty_fresh_prediction_not_replaced_by_consensus(tmp_path):
    row = attempt(prediction="", consensus_prediction="WI1234A")
    session = load(make_bundle(tmp_path / "source.alprsession", attempts=[row]), tmp_path)
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    session.annotate("attempt", "a1", plate_visibility="visible", is_plate=True)
    stats = session.statistics()
    assert stats["summary"]["no_reads"] == 1
    assert stats["summary"]["subject_success_rate"] == 1
    assert stats["subjects"][0]["consensus_repaired_result"]


def test_time_to_first_exact_uses_one_monotonic_clock_and_full_attempts(tmp_path):
    rows = [attempt("a1", status="NO_DETECTION", prediction=""), attempt("a2")]
    rows[1].update(attempt_started_elapsed_nanos=1_300_000_000, capture_source="auto_zoom", camera_zoom_ratio=2)
    session = load(make_bundle(tmp_path / "source.alprsession", samples=[], attempts=rows), tmp_path)
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    result = session.statistics()["subjects"][0]
    assert result["attempts_to_first_exact"] == 2 and result["time_to_first_exact_ms"] == 300
    assert result["exact_only_after_az"] and not result["baseline_exact_before_az"]
    assert result["first_exact_camera_zoom_ratio"] == 2


def test_completed_requires_gt_and_attempt_decisions(tmp_path):
    session = load(make_bundle(tmp_path / "source.alprsession", attempts=[attempt()]), tmp_path)
    with pytest.raises(ValueError):
        session.complete()
    key = next(iter(session.subjects))
    with pytest.raises(ValueError):
        session.set_subject(key, ground_truth=" - ")
    session.set_subject(key, ground_truth="WI1234A")
    with pytest.raises(ValueError):
        session.complete()
    session.annotate("attempt", "a1", plate_visibility="uncertain")
    session.annotate_invocation("a1", visible_plate_count="uncertain")
    session.complete()
    session.set_subject(key, ground_truth="NEW")
    assert session.review.review_status == "IN_PROGRESS"


def test_completed_ranking_quality_export_and_reopening(tmp_path):
    session = load(make_bundle(tmp_path / "source.alprsession", samples=[dict(capture_id="c1", prediction="ABCD")]), tmp_path)
    store = MobilePackageExperimentStore(tmp_path / "ranking")
    store.add_report(session.bundle.report)
    original = store.reports[0]
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="A")
    assert not store.apply_human_review(session)
    assert store.reports[0] == original
    session.complete()
    assert store.apply_human_review(session)
    reviewed = store.reports[0]
    assert reviewed.quality["cer"] == 3
    assert score_mobile_report(reviewed).metrics["cer"] == 3
    assert score_mobile_report(reviewed).quality == 0
    assert reviewed.memory == original.memory and reviewed.latency == original.latency
    assert reviewed.source_archive_sha256 == session.review.source_archive_sha256
    paths = session.export(tmp_path / "export")
    assert {path.name for path in paths} == {"review.json", "summary.csv", "subjects.csv", "character_confusion.csv"}
    result = json.loads(paths[0].read_text(encoding="utf-8"))
    assert result["subjects"][key]["ground_truth"] == "A"
    assert result["provenance"]["model_provenance"]["plate"]["checkpoint_sha256"] == "abc"
    session.set_subject(key, ground_truth="ABCD")
    # Even a crash before refreshing the ranking cannot publish a stale revision.
    assert store._current_review_quality(reviewed).quality == original.quality
    store.apply_human_review(session)
    assert store.reports[0].quality == original.quality


def test_duplicate_ids_and_cross_subject_links_rejected(tmp_path):
    rows = [dict(capture_id="c1", subject_key="wrong", attempt_id="a1")]
    path = make_bundle(tmp_path / "bad.alprsession", samples=rows, attempts=[attempt()])
    with pytest.raises(ValueError, match="subject_key"):
        load(path, tmp_path)


def test_not_evaluable_and_cancelled_crops_not_model_errors(tmp_path):
    rows = [attempt(stale_or_cancelled=True)]
    session = load(make_bundle(tmp_path / "source.alprsession", attempts=rows), tmp_path)
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="OTHER")
    assert session.statistics()["summary"]["evaluable_reads"] == 0
    assert session.statistics()["summary"]["evaluable_subjects"] == 0
    session.set_subject(key, evaluable=False)
    assert session.statistics()["summary"]["evaluable_subjects"] == 0


def test_mz_not_run_is_not_no_read(tmp_path):
    row = attempt(prediction="")
    row["mz_status"] = "NOT_RUN"
    session = load(make_bundle(tmp_path / "source.alprsession", attempts=[row]), tmp_path)
    session.set_subject(next(iter(session.subjects)), ground_truth="ABC")
    assert session.statistics()["summary"]["evaluable_reads"] == 0
    assert session.statistics()["summary"]["no_reads"] == 0


def test_s3_and_s4_character_statistics(tmp_path):
    predictions = ["ABXDEFGHIJ"] * 4 + ["ABCDEFGHJ"] * 3 + ["ABCDEFGHIJJ"] * 2 + ["ABCDEFGHIJ"]
    samples = [dict(capture_id=f"c{i}", subject_key="s/1", prediction=pred) for i, pred in enumerate(predictions)]
    session = load(make_bundle(tmp_path / "source.alprsession", samples=samples), tmp_path)
    session.set_subject("s/1", ground_truth="ABCDEFGHIJ")
    summary = session.statistics()["summary"]
    assert summary["gt_characters"] == 100
    assert (summary["incorrect_characters"], summary["missing_characters"], summary["extra_characters"]) == (4, 3, 2)
    assert summary["cer"] == .09
    samples = [dict(capture_id=f"d{i}", subject_key="s/2", prediction=pred) for i, pred in enumerate(["O1", "O1", "0I"])]
    session = MobileReviewSession.open(make_bundle(tmp_path / "confusion.alprsession", samples=samples), sidecar_path=tmp_path / "confusion.review.json")
    session.set_subject("s/2", ground_truth="01")
    assert session.statistics()["character_confusion"] == [dict(ground_truth="0", prediction="O", count=2), dict(ground_truth="1", prediction="I", count=1)]


def test_cer_above_one_preserved_for_imported_reports(tmp_path):
    from dataclasses import replace
    report = read_mobile_report_bundle(make_bundle(tmp_path / "source.alprsession")).report
    score = score_mobile_report(replace(report, quality={"cer": 1.5, "exact_match_rate": 0}))
    assert score.metrics["cer"] == 1.5 and score.quality == 0


def test_missing_evidence_must_be_explicitly_excluded(tmp_path):
    row = attempt()
    row["evidence_entry"] = ""
    session = load(make_bundle(tmp_path / "source.alprsession", attempts=[row], samples=[]), tmp_path)
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    session.annotate("attempt", "a1", plate_visibility="visible", is_plate=True)
    assert session.statistics()["summary"]["evaluable_mt_attempts"] == 0
    session.annotate_invocation("a1", visible_plate_count="one")
    with pytest.raises(ValueError, match="Brak dowodu"):
        session.complete()
    session.annotate("attempt", "a1", evaluable=False)
    session.annotate_invocation("a1", visible_plate_count="uncertain", evaluable=False)
    session.complete()


def test_session_metadata_and_empty_gt_reset(tmp_path):
    path = make_bundle(tmp_path / "source.alprsession", attempts=[attempt()])
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("session.json", json.dumps({"session_id": "s", "state": "ERROR", "collection_complete": False}))
    session = load(path, tmp_path)
    assert session.bundle.collection_session["state"] == "ERROR"
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="WI1234A")
    session.set_subject(key, ground_truth="", evaluable=None)
    assert session.statistics()["summary"]["evaluable_reads"] == 0
    assert session.completion_issues()


def test_stale_review_instance_cannot_overwrite_newer_decisions(tmp_path):
    path = make_bundle(tmp_path / "source.alprsession")
    session = load(path, tmp_path)
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="FIRST")
    stale = load(path, tmp_path)
    session.set_subject(key, ground_truth="SECOND")
    with pytest.raises(ValueError, match="innym oknie"):
        stale.set_subject(key, ground_truth="THIRD")
    assert load(path, tmp_path).review.subjects[key]["ground_truth"] == "SECOND"


def test_pending_attempts_visible_in_subject_progress(tmp_path):
    session = load(make_bundle(tmp_path / "source.alprsession", attempts=[attempt()]), tmp_path)
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    stats = session.statistics()
    assert stats["subjects"][0]["pending_decisions"] == 1
    assert not stats["subjects"][0]["reviewed"]
    assert stats["summary"]["not_reviewed_subjects"] == 1
    session.annotate("attempt", "a1", plate_visibility="visible", is_plate=True)
    stats = session.statistics()
    assert stats["subjects"][0]["pending_decisions"] == 1
    session.annotate_invocation("a1", visible_plate_count="one")
    stats = session.statistics()
    assert stats["subjects"][0]["pending_decisions"] == 0
    assert stats["summary"]["reviewed_subjects"] == 1
