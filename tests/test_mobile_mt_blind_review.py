import copy
import json

import pytest

from test_mobile_human_review import make_bundle, attempt, load
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking.mobile_package_experiments import MobilePackageExperimentStore, _file_sha256
from auto_annotation_tool.ranking.mobile_mt_invocations import group_mt_invocations


def children(count=1, *, invocation="mt-x", statuses=None):
    rows = []
    for index in range(max(1, count)):
        status = statuses[index] if statuses else "VALID_QUAD" if count else "NO_DETECTION"
        row = attempt(f"{invocation}-d{index}", status=status, prediction="WI1234A" if status == "VALID_QUAD" else "")
        row.update(mt_invocation_id=invocation, mt_detection_index=index if count else "", mt_detection_count=count,
                   scene_generation=1, visual_epoch=2, camera_transform_generation=3, source_sequence=10,
                   source_timestamp_nanos=123456789, roi_left=0, roi_top=0, roi_right=800, roi_bottom=600,
                   input_width=640, input_height=640, input_scale=.8, input_pad_x=0, input_pad_y=80,
                   mt_executed=True, mt_input_evidence_entry=f"samples/evidence/{invocation}-d0.jpg")
        rows.append(row)
    return rows


def session_for(tmp_path, rows):
    return load(make_bundle(tmp_path / "input.alprsession", attempts=rows, samples=[]), tmp_path)


@pytest.mark.parametrize("statuses,visible,truth,expected", [
    (["NO_DETECTION"], "one", [], (1, 0, 1, 0, 0, 0, 0, 0.0)),
    (["VALID_QUAD"], "one", [True], (1, 1, 0, 0, 0, 0, 0, 1.0)),
    (["VALID_QUAD"] * 3, "one", [True, True, True], (1, 1, 0, 0, 0, 0, 0, 1.0)),
    (["VALID_QUAD", "VALID_QUAD"], "one", [True, False], (1, 1, 0, 0, 0, 0, 1, 1.0)),
    (["DETECTION_INVALID_QUAD"], "one", [True], (1, 0, 0, 1, 0, 0, 0, 0.0)),
    (["VALID_QUAD"] * 2, "multiple", [True, True], (0, 0, 0, 0, 1, 0, 0, None)),
    (["VALID_QUAD"], "none", [False], (0, 0, 0, 0, 0, 0, 1, None)),
    (["VALID_QUAD"], "uncertain", [None], (0, 0, 0, 0, 0, 1, 0, None)),
])
def test_t1_to_t7_invocation_counts(tmp_path, statuses, visible, truth, expected):
    count = 0 if statuses == ["NO_DETECTION"] else len(statuses)
    rows = children(count, statuses=statuses)
    session = session_for(tmp_path, rows)
    session.annotate_invocation("mt-x", visible_plate_count=visible)
    for row, is_plate in zip(rows, truth):
        session.annotate("attempt", row["attempt_id"], is_plate=is_plate)
    stats = session.statistics()
    keys = ("evaluable_mt_invocations", "mt_successful_invocations", "mt_no_detection_invocations", "mt_invalid_quad_invocations",
            "mt_multi_plate_invocations", "mt_uncertain_invocations", "mt_false_detections", "mt_localization_success_rate")
    assert tuple(stats["summary"][key] for key in keys) == expected
    assert len(stats["mt_invocations"]) == 1
    assert stats["summary"]["mt_invocation_count"] == 1
    assert stats["summary"]["evaluable_mt_attempts"] == expected[0]  # compatibility alias


def test_t8_any_cancelled_child_excludes_entire_invocation(tmp_path):
    rows = children(2)
    rows[1]["stale_or_cancelled"] = True
    session = session_for(tmp_path, rows)
    session.annotate_invocation("mt-x", visible_plate_count="one")
    session.annotate("attempt", rows[0]["attempt_id"], is_plate=True)
    session.annotate("attempt", rows[1]["attempt_id"], is_plate=False)
    stats = session.statistics()["summary"]
    assert stats["evaluable_mt_invocations"] == stats["mt_false_detections"] == 0


def test_group_spans_subjects_and_never_uses_ocr(tmp_path):
    rows = children(2)
    rows[1]["subject_key"] = "s/sg-1/entity-2"
    rows += children(1, invocation="mt-y")
    session = session_for(tmp_path, rows)
    assert len(session.subjects) == 2
    assert len(session.mt_invocations) == 2
    assert len(session.mt_invocations["mt-x"].subject_keys) == 2
    for key in session.mt_invocations:
        session.annotate_invocation(key, visible_plate_count="one")
    assert session.statistics()["summary"]["evaluable_mt_invocations"] == 2


@pytest.mark.parametrize("field", ["session_id", "scene_generation", "visual_epoch", "camera_transform_generation",
    "source_sequence", "source_timestamp_nanos", "roi_left", "roi_right", "input_width", "input_scale", "input_pad_y"])
def test_inconsistent_invocation_identity_rejected(field):
    rows = children(2)
    rows[1][field] = "different" if field == "session_id" else 42
    records = {row["attempt_id"]: dict(row, id=row["attempt_id"]) for row in rows}
    with pytest.raises(ValueError, match=field):
        group_mt_invocations(records, "s")


@pytest.mark.parametrize("change", ["duplicate_index", "missing_index", "wrong_count", "missing_child", "zero_with_index"])
def test_invalid_detection_cardinality_rejected(tmp_path, change):
    rows = children(2)
    if change == "duplicate_index": rows[1]["mt_detection_index"] = 0
    if change == "missing_index": rows[1]["mt_detection_index"] = ""
    if change == "wrong_count": rows[1]["mt_detection_count"] = 3
    if change == "missing_child": rows.pop()
    if change == "zero_with_index":
        rows = children(0)
        rows[0]["mt_detection_index"] = 0
    with pytest.raises(ValueError):
        session_for(tmp_path, rows)


def test_old_invocation_without_detection_ordinals_and_attempt_fallback(tmp_path):
    rows = children(2)
    for row in rows:
        del row["mt_detection_index"], row["mt_detection_count"]
    session = session_for(tmp_path, rows)
    assert len(session.mt_invocations) == 1
    for row in rows:
        row.pop("mt_invocation_id")
    other = MobileReviewSession.open(make_bundle(tmp_path / "fallback.alprsession", attempts=rows, samples=[]), sidecar_path=tmp_path / "other.review.json")
    assert len(other.mt_invocations) == 2
    assert all(group.legacy_identity for group in other.mt_invocations.values())


def test_crop_is_not_evidence_for_whole_input(tmp_path):
    row = children()[0]
    row["mt_input_evidence_entry"] = ""
    row["evidence_entry"] = "samples/crops/roi.jpg"
    session = session_for(tmp_path, [row])
    session.annotate_invocation("mt-x", visible_plate_count="one")
    session.annotate("attempt", row["attempt_id"], is_plate=True)
    assert session.statistics()["summary"]["evaluable_mt_invocations"] == 0
    assert "Brak dowodu" in session.invocation_completion_issues("mt-x")[0]


def test_b1_b2_draft_autosave_does_not_confirm_or_reveal_gt(tmp_path):
    path = make_bundle(tmp_path / "source.alprsession")
    session = load(path, tmp_path)
    assert session.review.review_mode == "blinded_gt_v1"
    key = next(iter(session.subjects))
    session.save_subject_draft(key, ground_truth="WI1234A")
    assert not session.predictions_visible(key)
    assert not session.review.subjects[key].get("ground_truth")
    assert session.statistics()["summary"]["evaluable_reads"] == 0
    reopened = load(path, tmp_path)
    assert reopened.subject_draft(key) == "WI1234A"
    assert not reopened.predictions_visible(key)
    with pytest.raises(ValueError): reopened.complete()
    reopened.set_subject(key, ground_truth=reopened.subject_draft(key))
    assert reopened.predictions_visible(key)
    assert reopened.statistics()["summary"]["exact_reads"] == 1
    reopened.save_subject_draft(key, ground_truth="WI1234B")
    assert reopened.statistics()["summary"]["incorrect_reads"] == 1
    assert reopened.predictions_visible(key)


def test_review_mode_is_explicit_and_locked_after_first_saved_decision(tmp_path):
    session = load(make_bundle(tmp_path / "source.alprsession"), tmp_path)
    session.set_review_mode("assisted")
    key = next(iter(session.subjects))
    assert session.predictions_visible(key)
    session.set_review_mode("blinded_gt_v1")
    session.save_subject_draft(key, ground_truth="WI1234A")
    with pytest.raises(ValueError, match="stały"):
        session.set_review_mode("assisted")


def test_b4_legacy_completed_sidecar_opens_assisted_without_rewriting(tmp_path):
    session = session_for(tmp_path, [attempt()])
    key = next(iter(session.subjects))
    data = session.review.to_dict()
    for name in ("review_mode", "review_mode_locked", "mt_review_policy", "invocation_annotations"):
        data.pop(name)
    data.update(review_status="COMPLETED", review_revision=3, subjects={key: {"ground_truth": "WI1234A", "evaluable": True}},
                attempt_annotations={"a1": {"plate_visibility": "visible", "is_plate": True}})
    session.sidecar_path.write_text(json.dumps(data), encoding="utf-8")
    before = session.sidecar_path.read_bytes()
    reopened = load(session.path, tmp_path)
    assert reopened.review.review_mode == "assisted" and reopened.review.review_status == "COMPLETED"
    assert reopened.statistics()["summary"]["mt_localization_success_rate"] is None
    assert reopened.statistics()["summary"]["mt_uncertain_invocations"] == 1
    assert session.sidecar_path.read_bytes() == before


def test_legacy_multi_detection_visibility_is_not_guessed_as_one(tmp_path):
    rows = children(2)
    session = session_for(tmp_path, rows)
    data = session.review.to_dict()
    for field in ("review_mode", "review_mode_locked", "mt_review_policy", "invocation_annotations"):
        data.pop(field)
    data["attempt_annotations"] = {row["attempt_id"]: {"plate_visibility": "visible", "is_plate": True} for row in rows}
    session.sidecar_path.write_text(json.dumps(data), encoding="utf-8")
    old = load(session.path, tmp_path)
    assert old.statistics()["summary"]["evaluable_mt_invocations"] == 0
    assert old.invocation_completion_issues("mt-x")


def test_b5_b7_export_quality_mode_and_immutable_source(tmp_path):
    rows = children(2, statuses=["VALID_QUAD", "VALID_QUAD"])
    session = session_for(tmp_path, rows)
    source_hash = _file_sha256(session.path)
    for key in session.subjects:
        session.set_subject(key, ground_truth="WI1234A")
    session.annotate_invocation("mt-x", visible_plate_count="one")
    for row in rows:
        session.annotate("attempt", row["attempt_id"], is_plate=True)
    session.complete()
    store = MobilePackageExperimentStore(tmp_path / "ranking")
    store.add_report(session.bundle.report)
    store.apply_human_review(session)
    assert store.reports[0].quality["review_mode"] == "blinded_gt_v1"
    assert store.reports[0].quality["evaluable_mt_invocations"] == 1
    assert store.reports[0].latency == session.bundle.report.latency
    session.export(tmp_path / "export")
    payload = json.loads((tmp_path / "export/review.json").read_text(encoding="utf-8"))
    assert payload["invocation_annotations"]["mt-x"]["visible_plate_count"] == "one"
    assert payload["review_mode"] == "blinded_gt_v1"
    header = (tmp_path / "export/summary.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "review_mode" in header and "evaluable_mt_invocations" in header
    assert _file_sha256(session.path) == source_hash


def test_invocation_decisions_do_not_change_mz_or_alpr_metrics(tmp_path):
    rows = children(2)
    rows[0].update(prediction="", consensus_prediction="WI1234A", capture_source="auto_zoom", camera_zoom_ratio=2)
    rows[1]["prediction"] = "WI12B4A"
    samples = [dict(capture_id=f"c{i}", attempt_id=row["attempt_id"], subject_key=row["subject_key"]) for i, row in enumerate(rows)]
    session = load(make_bundle(tmp_path / "source.alprsession", samples=samples, attempts=rows), tmp_path)
    session.set_subject(next(iter(session.subjects)), ground_truth="WI1234A")
    before = session.statistics()
    for visible in ("one", "none", "multiple", "uncertain"):
        session.annotate_invocation("mt-x", visible_plate_count=visible)
        after = session.statistics()
        for key in ("exact_reads", "incorrect_reads", "no_reads", "cer", "gt_characters", "correct_characters",
                    "incorrect_characters", "missing_characters", "extra_characters", "subject_success_rate",
                    "median_time_to_first_exact_ms", "median_attempts_to_first_exact"):
            assert after["summary"][key] == before["summary"][key]
        for key in ("subject_exact_success", "attempts_to_first_exact", "time_to_first_exact_ms", "first_exact_capture_source",
                    "first_exact_camera_zoom_ratio", "baseline_exact_before_az", "exact_only_after_az", "consensus_repaired_result"):
            assert after["subjects"][0][key] == before["subjects"][0][key]
        assert after["reads"] == before["reads"] and after["character_confusion"] == before["character_confusion"]
