"""Truthful pair identity/qualification in presentation, with immutable metrics."""
import copy

import pytest

from test_mobile_p2b2_canvas_review import source_session, text_all
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_presentation import (
    execution_state, focused_assessment, geometry_assessment, plate_review_state,
)

GID = "mt-one"


def model(s, oid="", **kwargs):
    return focused_assessment(core.scene(s, GID), oid, execution_state(s, GID), **kwargs)


def test_fp_and_real_plate_have_distinct_pairs_and_geometry_domains(tmp_path):
    s = source_session(tmp_path); text_all(s)
    core.toggle_error(s, GID, "GT:a1"); core.check_scene(s, GID)
    before = s.review.to_dict(); summary = core.summary(s)
    good, fp = model(s, "GT:a0"), model(s, "GT:a1")
    assert good["geometry_pair"] == dict(object_id="GT:a0", parent_id="MT:a0", attempt_id="a0",
        mt_invocation_id=GID, image_entry="samples/frames/source.jpg")
    assert (good["polygon_iou_pct"], good["nmce_pct"]) == (100, 0)
    assert fp["geometry_pair"] is None and fp["geometry_status"] == "FP"
    assert fp["frame"] == "MT 2 — błędne wykrycie (FP)"
    assert fp["geometry"] == "IoU: nie dotyczy · NMCE: nie dotyczy"
    assert fp["polygon_iou_pct"] is fp["nmce_pct"] is fp["cer"] is None
    assert "FP" in fp["mz"] and "historia" in fp["gt"]
    assert not fp["gt_editable"] and good["gt_editable"]
    assert "1 błędne MT (FP)" in good["image_frames"]
    assert "Decyzja FP uwzględniona" in fp["summary_note"]
    assert s.review.to_dict() == before and core.summary(s) == summary
    assert summary["geometry_n"] == summary["evaluated_reads"] == summary["false_positive_mt"] == 1


def test_auto_preview_names_only_real_mt_without_mutating_selection_or_review(tmp_path):
    s = source_session(tmp_path)
    core.toggle_error(s, GID, "GT:a1")
    before = s.review.to_dict()
    result = model(s)
    assert result["focus_mode"] == "AUTO" and result["focus_id"] == "GT:a0"
    assert result["focus"] == "MT 1 · podgląd bez zaznaczenia"
    assert result["geometry_status"] == "UNCONFIRMED"
    assert result["geometry_pair"] is None and result["polygon_iou_pct"] is result["nmce_pct"] is None
    assert "Jeszcze nie oceniono geometrii" == result["geometry"]
    assert "niepotwierdzona" in result["reference"]
    assert core.summary(s)["geometry_n"] == 0 and s.review.to_dict() == before


def test_seed_copy_not_reference_until_explicit_acceptance_and_reset_invalidates_it(tmp_path):
    s = source_session(tmp_path, n=1)
    text_all(s)
    assert model(s)["polygon_iou_pct"] is None  # Entering text does not accept geometry.
    core.check_scene(s, GID)
    result = model(s)
    assert result["geometry_status"] == "ASSESSED"
    assert result["reference"] == "Referencja: oryginał MT 1 zaakceptowany · ASSISTED"
    assert (result["polygon_iou_pct"], result["nmce_pct"]) == (100, 0)
    core.remove_operator_geometry(s, GID, "GT:a0")
    assert model(s)["geometry_status"] == "UNCONFIRMED" and model(s)["polygon_iou_pct"] is None
    assert core.summary(s)["geometry_n"] == 0


def test_corrected_reference_matches_canonical_pair_and_never_changes_summary_by_view(tmp_path):
    s = source_session(tmp_path, n=1); text_all(s)
    points = copy.deepcopy(core.scene(s, GID)["objects"][0]["quad"]); points[0][0] += 10
    core.correct_plate(s, GID, "GT:a0", points)
    draft = model(s)
    expected = core.compare_mt_pose_to_gt(core.scene(s, GID)["parents"][0]["quad"], points, 1000, 700)
    assert draft["draft"] and draft["polygon_iou_pct"] == expected["polygon_iou_pct"]
    assert core.summary(s)["geometry_n"] == 0
    core.check_scene(s, GID)
    summary = core.summary(s); before = s.review.to_dict()
    assessed = model(s, "GT:a0")
    row = summary["rows"][0]
    for key in ("polygon_iou_pct", "nmce_pct"):
        assert assessed[key] == draft[key] == row[key]
    assert assessed["frame_state"] == "MT_CORRECTED"
    assert assessed["reference"] == "Ramka operatora · powiązana z MT 1 · ASSISTED"
    assert assessed["geometry_pair"]["object_id"] == row["object_id"]
    assert assessed["geometry_pair"]["parent_id"] == row["parent_id"]
    assert s.review.to_dict() == before


@pytest.mark.parametrize("fault", ["duplicate_parent", "duplicate_child", "wrong_attempt", "missing_parent", "wrong_frame"])
def test_ambiguous_identity_cannot_borrow_reference_gt_or_metrics(tmp_path, fault):
    s = source_session(tmp_path, n=1); text_all(s); core.check_scene(s, GID)
    scene = copy.deepcopy(core.scene(s, GID)); e = execution_state(s, GID)
    if fault == "duplicate_parent": scene["parents"].append(copy.deepcopy(scene["parents"][0]))
    elif fault == "duplicate_child":
        child = copy.deepcopy(scene["objects"][0]); child["object_id"] = "different-object"
        scene["objects"].append(child)
    elif fault == "wrong_attempt": scene["objects"][0]["attempt_id"] = "wrong"
    elif fault == "missing_parent": scene["objects"][0]["parent_id"] = "missing"
    else: e = dict(e, image_entry="other.jpg")
    before = copy.deepcopy(scene)
    result = focused_assessment(scene, "GT:a0", e)
    assert result["state"] == "UNBOUND"
    assert result["geometry_pair"] is None
    assert result["polygon_iou_pct"] is result["nmce_pct"] is result["cer"] is None
    assert result["identity_reason"] and not result["gt_editable"]
    assert scene == before


def test_multiple_real_plates_and_extra_fp_require_explicit_focus(tmp_path):
    s = source_session(tmp_path, n=3); text_all(s)
    core.toggle_error(s, GID, "GT:a2"); core.check_scene(s, GID)
    before = core.summary(s)
    assert model(s)["focus_mode"] == "NONE" and model(s)["polygon_iou_pct"] is None
    for index in range(3):
        result = model(s, f"GT:a{index}")
        assert result["focus"].startswith(f"MT {index+1}")
        if index < 2:
            assert result["gt"] == f"GT: WA{index}1234"
            assert result["geometry_pair"]["attempt_id"] == f"a{index}"
        else:
            assert result["geometry_status"] == "FP" and result["geometry_pair"] is None
    assert core.summary(s) == before
    assert (before["linked_plates"], before["false_positive_mt"], before["geometry_n"], before["evaluated_reads"]) == (2, 1, 2, 2)


def test_live_reference_is_explicitly_unsaved_and_fp_never_gets_a_live_metric(tmp_path):
    s = source_session(tmp_path, n=1)
    scene = core.scene(s, GID); obj = scene["objects"][0]; before = s.review.to_dict()
    points = copy.deepcopy(obj["quad"]); points[0][0] += 9
    live = geometry_assessment(scene, obj, execution_state(s, GID), live_quad=points)
    assert live["geometry_status"] == "LIVE" and "niezapisana" in live["reference"]
    assert live["polygon_iou_pct"] < 100 and live["nmce_pct"] > 0
    assert s.review.to_dict() == before and core.summary(s)["geometry_n"] == 0
    core.toggle_error(s, GID, "GT:a0")
    scene = core.scene(s, GID)
    fp = geometry_assessment(scene, scene["objects"][0], execution_state(s, GID), live_quad=points)
    assert fp["geometry_status"] == "FP" and fp["polygon_iou_pct"] is fp["nmce_pct"] is None


def test_fp_draft_and_confirmed_image_do_not_imply_formal_session_approval(tmp_path):
    s = source_session(tmp_path); text_all(s)
    core.toggle_error(s, GID, "GT:a1")
    assert model(s, "GT:a1")["summary_note"].startswith("Wstępna decyzja FP")
    assert core.summary(s)["false_positive_mt"] == 0
    core.check_scene(s, GID)
    assert core.summary(s)["false_positive_mt"] == 1
    assert s.review.review_status == "IN_PROGRESS"
    before = s.review.to_dict()
    for oid in ["", "GT:a0", "GT:a1"]: model(s, oid)
    assert s.review.to_dict() == before
