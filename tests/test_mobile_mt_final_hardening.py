import copy
import csv
import json

import pytest

from test_mobile_mt_blind_review import children, session_for
from test_mobile_human_review import make_bundle, load
from auto_annotation_tool.ranking.mobile_mt_invocations import detection_box_on_mt_input
from auto_annotation_tool.ranking.mobile_package_experiments import MobilePackageExperimentStore, _file_sha256


@pytest.mark.parametrize("count,status", [(0, "NO_DETECTION"), (3, "VALID_QUAD"), (2, "DETECTION_INVALID_QUAD")])
def test_execution_error_excludes_whole_invocation_from_quality(tmp_path, count, status):
    rows = children(count, statuses=[status] * max(1, count))
    rows[-1]["execution_error"] = "  RuntimeException  "
    session = session_for(tmp_path, rows)
    session.annotate_invocation("mt-x", visible_plate_count="one")
    for index, row in enumerate(rows):
        session.annotate("attempt", row["attempt_id"], is_plate=index == 0)
    group = session.mt_invocations["mt-x"]
    assert group.execution_failed and group.execution_errors == ("RuntimeException",)
    stats = session.statistics()
    assert stats["summary"]["mt_execution_error_invocations"] == 1
    for key in ("evaluable_mt_invocations", "mt_successful_invocations", "mt_no_detection_invocations",
                "mt_invalid_quad_invocations", "mt_false_detections", "mt_multi_plate_invocations", "mt_uncertain_invocations"):
        assert stats["summary"][key] == 0
    assert stats["summary"]["mt_localization_success_rate"] is None
    result, = stats["mt_invocations"]
    assert result["outcome"] == "execution_error" and not result["included"]
    assert result["execution_errors"] == ["RuntimeException"]
    assert len(result["records"]) == max(1, count)
    assert result["false_detections"] == 0


@pytest.mark.parametrize("cancelled,executed,outcome,counter", [
    (True, True, "cancelled", 0), (True, False, "cancelled", 0),
    (False, False, "not_executed", 0), (False, True, "execution_error", 1),
])
def test_error_precedence_and_completion_without_mt_decisions(tmp_path, cancelled, executed, outcome, counter):
    rows = children(0)
    rows[0].update(execution_error="backend failed", stale_or_cancelled=cancelled, mt_executed=executed)
    session = session_for(tmp_path, rows)
    assert session.invocation_completion_issues("mt-x") == []
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    session.complete()
    assert not session.review.invocation_annotations and not session.review.attempt_annotations
    stats = session.statistics()
    assert stats["mt_invocations"][0]["outcome"] == outcome
    assert stats["summary"]["mt_execution_error_invocations"] == counter


def test_all_distinct_errors_preserved_and_failed_children_need_no_decisions(tmp_path):
    rows = children(3)
    for row, error in zip(rows, ("  backend failed  ", "decode failed\ncontext", "backend failed")):
        row["execution_error"] = error
    session = session_for(tmp_path, rows)
    assert session.mt_invocations["mt-x"].execution_errors == ("backend failed", "decode failed\ncontext")
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    session.complete()
    assert not session.review.attempt_annotations and not session.review.invocation_annotations


@pytest.mark.parametrize("error", [None, "", "  \t\n"])
def test_legacy_or_empty_error_is_not_failure(tmp_path, error):
    rows = children(0)
    if error is not None:
        rows[0]["execution_error"] = error
    session = session_for(tmp_path, rows)
    group = session.mt_invocations["mt-x"]
    assert not group.execution_failed and group.execution_errors == ()
    session.annotate_invocation("mt-x", visible_plate_count="one")
    assert session.statistics()["summary"]["mt_no_detection_invocations"] == 1
    assert session.statistics()["summary"]["mt_execution_error_invocations"] == 0


def test_error_counter_export_publication_and_source_immutability(tmp_path):
    rows = children(0)
    rows[0]["execution_error"] = "backend failed"
    session = session_for(tmp_path, rows)
    source_hash = _file_sha256(session.path)
    store = MobilePackageExperimentStore(tmp_path / "ranking")
    store.add_report(session.bundle.report)
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    store.apply_human_review(session)
    assert store.reports[0].quality.get("quality_source") != "human_review"
    session.complete()
    store.apply_human_review(session)
    assert store.reports[0].quality["mt_execution_error_invocations"] == 1
    assert store.reports[0].latency == session.bundle.report.latency
    assert store.reports[0].memory == session.bundle.report.memory
    session.export(tmp_path / "export")
    payload = json.loads((tmp_path / "export/review.json").read_text(encoding="utf-8"))
    assert payload["derived_metrics"]["summary"]["mt_execution_error_invocations"] == 1
    assert not payload["invocation_annotations"]
    with (tmp_path / "export/summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.DictReader(handle))["mt_execution_error_invocations"] == "1"
    assert load(session.path, tmp_path).review.review_status == "COMPLETED"
    assert _file_sha256(session.path) == source_hash


def test_error_does_not_change_mz_cer_or_alpr(tmp_path):
    rows = children(2)
    rows[0].update(prediction="", consensus_prediction="WI1234A", capture_source="auto_zoom", camera_zoom_ratio=2)
    rows[1]["prediction"] = "WI12B4A"
    samples = [dict(capture_id=f"c{i}", attempt_id=row["attempt_id"], subject_key=row["subject_key"]) for i, row in enumerate(rows)]
    outcomes = []
    for failed in (False, True):
        directory = tmp_path / str(failed)
        directory.mkdir()
        data = copy.deepcopy(rows)
        if failed:
            data[1]["execution_error"] = "decode failed"
        session = load(make_bundle(directory / "input.alprsession", attempts=data, samples=samples), directory)
        session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
        session.annotate_invocation("mt-x", visible_plate_count="one")
        for row in data:
            session.annotate("attempt", row["attempt_id"], is_plate=True)
        outcomes.append(session.statistics())
    before, after = outcomes
    for key in ("exact_reads", "incorrect_reads", "no_reads", "cer", "gt_characters", "correct_characters",
                "incorrect_characters", "missing_characters", "extra_characters", "subject_success_rate",
                "median_time_to_first_exact_ms", "median_attempts_to_first_exact"):
        assert after["summary"][key] == before["summary"][key]
    assert after["reads"] == before["reads"] and after["character_confusion"] == before["character_confusion"]
    assert after["subjects"] == before["subjects"]


def geometry(**changes):
    return dict(plate_left=100, plate_top=50, plate_right=300, plate_bottom=150,
                roi_left=0, roi_top=0, input_scale=1, input_pad_x=0, input_pad_y=0,
                input_width=640, input_height=640) | changes


@pytest.mark.parametrize("changes,expected", [
    ({}, (100, 50, 300, 150)),
    (dict(roi_left=200, roi_top=100, input_scale=2, input_pad_y=20,
          plate_left=250, plate_top=120, plate_right=350, plate_bottom=170), (100, 60, 300, 160)),
    (dict(plate_left=-30, plate_top=-2, plate_right=700, plate_bottom=650), (0, 0, 640, 640)),
    (dict(plate_left=700, plate_right=800), None),
    (dict(plate_right=100), None), (dict(plate_bottom=50), None),
    (dict(input_scale=0), None), (dict(input_scale=-1), None),
    (dict(input_width=0), None), (dict(input_height=-2), None),
    (dict(plate_left=float("nan")), None), (dict(input_scale="inf"), None),
    (dict(input_scale=1e308), None), (dict(input_pad_x=True), None),
    (dict(input_pad_y="bad"), None),
])
def test_detection_mapping_r0_roi_clipping_and_invalid_values(changes, expected):
    assert detection_box_on_mt_input(geometry(**changes)) == expected


@pytest.mark.parametrize("missing", list(geometry()))
def test_detection_mapping_never_guesses_missing_fields(missing):
    row = geometry()
    del row[missing]
    assert detection_box_on_mt_input(row) is None
