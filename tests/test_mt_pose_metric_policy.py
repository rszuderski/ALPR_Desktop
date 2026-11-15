from copy import deepcopy
import hashlib
import json

import pytest

from auto_annotation_tool.ranking import mt_geometry_review as gate
from auto_annotation_tool.ranking.mt_metric_policy import load_metric_policy, bound_pose_exclusion
from pose478_fixture import cohort
from test_mt_geometry_review import review


@pytest.fixture
def scenes():
    return cohort({"file": "fixture.jpg", "sha256": hashlib.sha256(b"fixture").hexdigest(), "domain": "DAY",
                   "image": {"oriented_size": [2048, 1153], "exif_orientation": 1},
                   "plates": [{"bbox": [10, 20, 100, 50], "quad": [[10, 20], [100, 20], [100, 50], [10, 50]]}]})


def excluded_plate(scenes):
    return next((s, p) for s in scenes for p in s["plates"] if p["plate_id"] == "plate_000421")


def test_scope_counts_are_identical_for_all_three_models_without_altering_gt(scenes):
    policy = load_metric_policy()
    originals = [(p["plate_id"], deepcopy(p["bbox"]), deepcopy(p["quad"])) for s in scenes for p in s["plates"]]
    results = []
    for model in policy["model_labels"]:
        result = gate.assess_metric_scope(scenes, policy)
        assert result["metric_counts"] == policy["expected_counts"]
        assert not result["blocking_geometry_issues"]
        assert len(result["geometry_issues"]) == 1
        assert len(result["metric_exclusions"]) == 1
        results.append(result)
    assert results[0] == results[1] == results[2]
    assert originals == [(p["plate_id"], p["bbox"], p["quad"]) for s in scenes for p in s["plates"]]
    scene, plate = excluded_plate(scenes)
    assert plate["bbox_eligible"] and not plate["pose_eligible"]
    assert plate["pose_evaluation_status"] == "EXCLUDED_BY_OPERATOR"
    assert plate["geometry_issue"] == "GT_CORNER_ORDER_REVERSED"
    assert sum(p["text_decision"] == "excluded_unreadable" for s in scenes for p in s["plates"]) == 30


@pytest.mark.parametrize("field", ["plate_id", "source_annotation_id", "bbox", "quad", "sha256", "file", "size", "domain"])
def test_exclusion_requires_exact_source_and_original_geometry(scenes, field):
    policy = load_metric_policy()
    scene, plate = excluded_plate(scenes)
    if field in ("plate_id", "source_annotation_id"):
        plate[field] += "_changed"
    elif field == "bbox":
        plate["bbox"][0] += .01
    elif field == "quad":
        plate["quad"][0][0] += .01
    elif field == "size":
        scene["image"]["oriented_size"][0] += 1
    elif field == "domain":
        scene["domain"] = "NIGHT"
    else:
        scene[field] += "_changed"
    result = gate.assess_metric_scope(scenes, policy)
    assert not result["metric_exclusions"]
    assert result["blocking_geometry_issues"]


@pytest.mark.parametrize("defect", ["bbox_outside", "bbox_degenerate", "missing_quad", "nonfinite", "bowtie", "different_plate", "quad_reordered"])
def test_exception_never_excuses_a_different_geometry_problem(scenes, defect):
    policy = load_metric_policy()
    scene, plate = excluded_plate(scenes)
    if defect == "bbox_outside":
        plate["bbox"][2] = 4000
    elif defect == "bbox_degenerate":
        plate["bbox"][2] = plate["bbox"][0]
    elif defect == "missing_quad":
        plate["quad"] = None
    elif defect == "nonfinite":
        plate["quad"][0][0] = float("nan")
    elif defect == "bowtie":
        plate["quad"][1], plate["quad"][2] = plate["quad"][2], plate["quad"][1]
    elif defect == "quad_reordered":
        plate["quad"] = [plate["quad"][i] for i in (0, 3, 2, 1)]
    else:
        scenes[0]["plates"][0]["quad"].reverse()
    result = gate.assess_metric_scope(scenes, policy)
    assert result["blocking_geometry_issues"]
    assert len(result["metric_exclusions"]) == (1 if defect == "different_plate" else 0)
    if defect.startswith("bbox"):
        assert not plate["bbox_eligible"]


def test_operator_authorization_does_not_approve_visual_protocol_or_inference():
    policy = load_metric_policy()
    assert not policy["authorization"]["visual_geometry_protocol_approved"]
    assert not policy["authorization"]["inference_authorized_by_this_record"]
    assert policy["metric_semantics"]["official_pose_map"].startswith("NOT_COMPUTED")
    assert policy["exclusions"][0]["corner_reordering"] == "forbidden"


def test_recheck_rejects_changed_eligibility_even_after_rehash(scenes, tmp_path):
    policy = load_metric_policy()
    scope = gate.assess_metric_scope(scenes, policy)
    report = {"schema": gate.REVIEW_SCHEMA, "status": "READY_FOR_REVIEW", "policy": deepcopy(gate.POLICY),
              "selection_sha": gate.SELECTION_SHA, "gt_freeze_sha": gate.FREEZE_SHA,
              "scenes": scenes, "models": [], "counts": {}, "review_scene_hashes": [], "runtime": {},
              "code_sha256": {}, "source_files": {}, "metric_policy": policy,
              "repo_root": str(tmp_path), "source_scene_root": str(tmp_path),
              "selection_dir": str(tmp_path / "selection"), "freeze_dir": str(tmp_path / "freeze"),
              "metric_policy_sha256": gate.METRIC_POLICY_SHA, **scope}
    excluded_plate(report["scenes"])[1]["bbox_eligible"] = False
    report["review_sha256"] = gate.review_fingerprint(report)
    with pytest.raises(ValueError, match="METRIC_ELIGIBILITY_CHANGED"):
        gate.recheck_review(report)


def test_changed_policy_or_duplicate_scene_fails_closed(scenes):
    policy = load_metric_policy()
    changed = deepcopy(policy)
    changed["exclusions"][0]["excluded_from"].append("box_detection_metrics")
    with pytest.raises(ValueError, match="METRIC_POLICY_CHANGED"):
        gate.assess_metric_scope(scenes, changed)
    scenes[-1]["sha256"] = scenes[0]["sha256"]
    assert any(p["reason"] == "METRIC_IDENTITY_NOT_UNIQUE" for p in gate.assess_metric_scope(scenes, policy)["blocking_geometry_issues"])


def test_old_approval_schema_is_rejected_before_use():
    with pytest.raises(ValueError, match="PREFLIGHT_SCHEMA_INVALID"):
        gate.recheck_review({"schema": "alpr.mt_geometry_preflight.v1", "status": "READY_FOR_REVIEW"})


def test_prepared_protocol_preserves_decision_but_does_not_approve_or_run(review, tmp_path):
    output = tmp_path.parent / (tmp_path.name + "_protocols")
    target = gate.prepare_geometry_protocol(review, output)
    status = json.loads((target / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "AWAITING_OPERATOR_GEOMETRY_REVIEW"
    assert status["measurement_status"] == "NOT_RUN" and not status["visual_gt_approved"]
    assert status["metric_counts"] == review["metric_counts"]
    assert not (target / "geometry_policy.json").exists()  # No forged operator signature.
    assert json.loads((target / "metric_eligibility_policy.json").read_text(encoding="utf-8")) == review["metric_policy"]
    assert json.loads((target / "preflight.json").read_text(encoding="utf-8"))["metric_exclusions"] == review["metric_exclusions"]


def test_acceptance_is_bound_to_all_review_scenes_and_the_exact_metric_scope(review, tmp_path):
    changed = deepcopy(review)
    changed["review_scene_hashes"] = changed["review_scene_hashes"][:-1]
    changed["review_sha256"] = gate.review_fingerprint(changed)
    with pytest.raises(ValueError, match="REQUIRED_VISUAL_REVIEW_CHANGED"):
        gate.recheck_review(changed)
    output = tmp_path.parent / (tmp_path.name + "_decisions")
    path, decision = gate.save_operator_decision(review, output, operator="TEST FIXTURE ONLY",
        reviewed_scenes=review["review_scene_hashes"], accepted=True)
    assert gate.require_current_approval(review, decision)
    assert decision["metric_counts"]["bbox"] == 479 and decision["metric_counts"]["pose"] == 478
    assert (path / "metric_eligibility_policy.json").exists()
    decision["metric_counts"]["bbox"] = 478
    decision["contract_sha256"] = gate.digest({k: v for k, v in decision.items() if k != "contract_sha256"})
    with pytest.raises(ValueError, match="METRIC_POLICY_APPROVAL_MISMATCH"):
        gate.require_current_approval(review, decision)
    decision["schema"] = "alpr.mt_geometry_operator_contract.v1"
    with pytest.raises(ValueError, match="EXIF_OPERATOR_APPROVAL_REQUIRED"):
        gate.require_current_approval(review, decision)


@pytest.mark.parametrize("field", ["quad", "bbox", "source_annotation_id", "text_decision", "scene_file"])
def test_rehashed_report_cannot_replace_the_frozen_source_values(review, field):
    scene = review["scenes"][0]
    plate = scene["plates"][0]
    if field == "quad":
        plate["quad"][0][0] += .01
    elif field == "bbox":
        plate["bbox"][0] += .01
    elif field == "scene_file":
        scene["file"] += "_changed"
    elif field == "text_decision":
        # Keep 449 OCR by exchanging two decisions; counts alone cannot detect this.
        other = next(p for s in review["scenes"] for p in s["plates"] if p["text_decision"] == "approved")
        plate["text_decision"], other["text_decision"] = other["text_decision"], plate["text_decision"]
    else:
        plate[field] += "_changed"
    review.update(gate.assess_metric_scope(review["scenes"], review["metric_policy"]))
    review["review_sha256"] = gate.review_fingerprint(review)
    with pytest.raises(ValueError, match="REPORT_DIFFERS_FROM_FROZEN_GT|REPORT_SCENE_IDENTITY_CHANGED"):
        gate.recheck_review(review)
