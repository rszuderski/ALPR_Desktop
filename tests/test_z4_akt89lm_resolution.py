"""Operator decisions on synthetic sources; production archives are never edited."""
import copy
import csv
import io
import json
import zipfile

import pytest

from test_mobile_p2b2_canvas_review import source_session, text_all, MANUAL
from test_mobile_human_review import load
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_presentation import (
    execution_state, focused_assessment, scene_result_status, image_frame_status,
)
from auto_annotation_tool.ranking.mobile_canvas_resolution import (
    link_manual_to_mt, exclude_missing_source_pose, missing_mt_geometry,
)

GID = "mt-one"


def invalid_source(tmp_path, *, missing_evidence=False, mz=False):
    original = source_session(tmp_path / "template", n=1, mz=mz)
    target = tmp_path / "invalid.alprsession"
    with zipfile.ZipFile(original.bundle.path) as src, zipfile.ZipFile(target, "w") as dest:
        for name in src.namelist():
            if name == 'telemetry/traces.jsonl':
                continue
            data = src.read(name)
            if name == "samples/attempts.csv":
                rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
                rows[0].update(mt_status="VALID_QUAD" if missing_evidence else "DETECTION_INVALID_QUAD",
                               mt_pose_status="" if missing_evidence else "INVALID_QUAD", mt_pose_corners_norm="",
                               plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
                out = io.StringIO(newline="")
                writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
                data = out.getvalue().encode("utf-8")
            dest.writestr(name, data)
        dest.writestr('telemetry/traces.jsonl', json.dumps(dict(source_sequence=17,source_timestamp_ns=100000,
            scene_generation=1,visual_epoch=1,camera_transform_generation=1,
            counters=dict(plate_detections_after_dedup=1))))
    session = load(target, tmp_path)
    session.set_review_mode("assisted")
    return session


def metrics(s):
    d = core.summary(s)
    return {k: d["false_positive_mt" if k == "false_positives" else k] for k in ("expected_plates", "linked_plates", "missed_plates", "false_positives",
        "geometry_n", "evaluated_reads", "gt_characters", "text_errors", "exact_reads")}


def test_explicit_same_plate_link_preserves_gt_parent_audit_and_denominators(tmp_path):
    s = source_session(tmp_path, n=1); text_all(s)
    manual = core.add_plate(s, GID, MANUAL)
    core.set_text(s, GID, manual, "HISTORY")  # legacy manual GT remains auditable, never transferred
    before = core.scene(s, GID)
    assert metrics(s)["geometry_n"] == 0  # unchecked
    link_manual_to_mt(s, GID, manual, "GT:a0")
    after = core.scene(s, GID)
    assert after["parents"] == before["parents"] and after["source_binding"] == before["source_binding"]
    assert len(after["objects"]) == 1 and after["objects"][0]["gt_text"] == "WA01234"
    assert after["objects"][0]["quad"] == before["objects"][1]["quad"]
    assert after["events"][-1]["object_before"]["manual"] == before["objects"][1]
    assert not after["checked"]
    core.check_scene(s, GID)
    assert metrics(s) == dict(expected_plates=1, linked_plates=1, missed_plates=0, false_positives=0,
        geometry_n=1, evaluated_reads=1, gt_characters=7, text_errors=0, exact_reads=1)
    reopened = load(s.bundle.path, tmp_path)
    assert core.scene(reopened, GID) == core.scene(s, GID)
    assert metrics(reopened) == metrics(s)


def test_independent_fp_and_fn_without_mz_or_gt(tmp_path):
    s = invalid_source(tmp_path)
    manual = core.add_plate(s, GID, MANUAL)
    core.toggle_error(s, GID, "GT:a0")
    core.check_scene(s, GID)
    assert metrics(s) == dict(expected_plates=1, linked_plates=0, missed_plates=1, false_positives=1,
        geometry_n=0, evaluated_reads=0, gt_characters=0, text_errors=0, exact_reads=0)
    row = next(r for r in core.summary(s)["rows"] if r["object_id"] == manual)
    assert row["cer"] is None and row["polygon_iou_pct"] is None and row["nmce_pct"] is None


def test_two_real_plates_one_detected_remain_two_without_explicit_link(tmp_path):
    s = source_session(tmp_path, n=1); text_all(s)
    manual = core.add_plate(s, GID, MANUAL)
    core.check_scene(s, GID)
    before = s.review.to_dict()
    model = focused_assessment(core.scene(s, GID), manual, execution_state(s, GID))
    assert model["cer"] is None and model["nmce_pct"] is None and not model["gt_editable"]
    assert metrics(s) == dict(expected_plates=2, linked_plates=1, missed_plates=1, false_positives=0,
        geometry_n=1, evaluated_reads=1, gt_characters=7, text_errors=0, exact_reads=1)
    assert s.review.to_dict() == before


def test_invalid_source_cannot_be_replaced_by_manual_pose_or_silently_counted(tmp_path):
    s = invalid_source(tmp_path)
    manual = core.add_plate(s, GID, MANUAL)
    core.set_scene_review_state(s, GID, checked=True)
    before = s.review.to_dict(); scene = core.scene(s, GID)
    assert len(missing_mt_geometry(scene)) == 1 and core._content_issues(scene)
    assert core.counts(scene)["expected"] == 1  # invalid MT is unresolved, not a second GT
    assert metrics(s)["expected_plates"] == metrics(s)["missed_plates"] == 0
    model = focused_assessment(scene, "", execution_state(s, GID))
    assert model["summary_note"] == "Sprawdzone — rozstrzygnij wskazanie MT"
    assert "MT bez poprawnej Pose" in image_frame_status(scene, execution_state(s, GID))
    # P0 now permits explicit object identity without replacing original Pose.
    link_manual_to_mt(s, GID, manual, "GT:a0")
    after = core.scene(s, GID)
    assert after["parents"] == scene["parents"] and after["parents"][0]["quad"] is None
    assert after["objects"][0]["identity_decision"]["pose_status"] == "INVALID"
    assert metrics(s)["linked_plates"] == 1 and metrics(s)["missed_plates"] == 0
    assert metrics(s)["geometry_n"] == metrics(s)["evaluated_reads"] == 0


def test_missing_pose_alone_cannot_exclude_image_or_erase_annotations(tmp_path):
    s = invalid_source(tmp_path)
    core.set_text(s, GID, "GT:a0", "SAVED")
    core.add_plate(s, GID, MANUAL)
    before = core.scene(s, GID)
    with pytest.raises(ValueError, match="Sam brak Pose"):
        exclude_missing_source_pose(s, GID)
    after = core.scene(s, GID)
    assert after == before
    assert not after["exclusion_reason"] and core.group_issues(s, GID)
    assert set(metrics(s).values()) == {0}
    reopened = load(s.bundle.path, tmp_path)
    assert core.scene(reopened, GID) == after
    assert reopened.review.review_revision == s.review.review_revision
    assert metrics(reopened) == metrics(s)


def test_multiplate_link_changes_only_explicit_target(tmp_path):
    s = source_session(tmp_path, n=2); text_all(s)
    manual = core.add_plate(s, GID, MANUAL)
    before = core.scene(s, GID)
    link_manual_to_mt(s, GID, manual, "GT:a1")
    after = core.scene(s, GID)
    assert after["objects"][0] == before["objects"][0]
    assert after["objects"][1]["gt_text"] == before["objects"][1]["gt_text"]
    core.check_scene(s, GID)
    assert metrics(s)["geometry_n"] == 2 and metrics(s)["gt_characters"] == 14
    assert metrics(s)["missed_plates"] == 0


def test_raw_nms_or_pending_cannot_accept_manual_link(tmp_path, monkeypatch):
    s = source_session(tmp_path, n=1, mz=False)
    manual = core.add_plate(s, GID, MANUAL)
    base = core.scene(s, GID); before = s.review.to_dict()
    for pending in (True, False):
        scene = copy.deepcopy(base)
        if pending:
            scene["_post_nms_pending"] = "a0"
        else:
            scene["objects"][0]["post_nms_suppressed"] = True
        monkeypatch.setattr(core, "scene", lambda *a: scene)
        with pytest.raises(ValueError):
            link_manual_to_mt(s, GID, manual, "GT:a0")
    assert s.review.to_dict() == before


def test_operator_copy_follows_confirmation_completeness_exclusion_and_undo_state(tmp_path):
    s = source_session(tmp_path, n=1); e = execution_state(s, GID)
    note = lambda: scene_result_status(core.scene(s, GID), e)
    assert note() == dict(summary_note="Wynik wstępny — jeszcze nie liczony do podsumowania", summary_tone="muted")
    core.set_scene_review_state(s, GID, checked=True)
    assert "Rozstrzygnij wskazanie na Canvas" in note()["summary_note"] and "uwzględniony" not in note()["summary_note"]
    text_all(s); core.check_scene(s, GID)
    assert note()["summary_note"] == "Wynik uwzględniony w podsumowaniu sesji"
    core.exclude_scene(s, GID, "Test")
    assert note()["summary_note"] == "Obraz wyłączony z oceny jakości"
    core.set_scene_review_state(s, GID, checked=False)
    assert note()["summary_tone"] == "muted"
    unbound = dict(e, image_entry="other.jpg")
    assert "uwzględniony" not in scene_result_status(core.scene(s, GID), unbound)["summary_note"]


def test_zero_detections_copy_does_not_invent_cer_or_fn(tmp_path):
    s = source_session(tmp_path, n=0, mz=False)
    model = focused_assessment(core.scene(s, GID), "", execution_state(s, GID))
    assert "MT nie wykrył tablicy. MZ nie uruchomiono — CER nie dotyczy" in model["summary_note"]
    assert model["cer"] is None and metrics(s)["missed_plates"] == 0


def test_invalid_pose_is_not_reported_as_zero_detections(tmp_path):
    s = invalid_source(tmp_path)
    model = focused_assessment(core.scene(s, GID), "", execution_state(s, GID))
    assert model["execution"]["detections"] == 1
    assert "MT nie wykrył tablicy." not in model["summary_note"]
