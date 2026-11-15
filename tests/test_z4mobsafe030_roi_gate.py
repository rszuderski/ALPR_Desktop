"""Z4MOBSAFE030: formal science must fail closed for historic operator GT."""
from __future__ import annotations
import json
from types import SimpleNamespace
from auto_annotation_tool.ranking.mobile_multi_plate_verification import (
    multi_scientific_summary, set_count,
)
from auto_annotation_tool.ranking.mobile_roi_evaluation import (
    invocation_roi_audit, formal_multi_plate_gate,
)
from test_z4mobrev024_multi_plate import MultiSession, populated


def _roi_session(tmp_path, *, policy="r0_full_frame", roi=(0, 0, 480, 640), n_pred=1):
    s = MultiSession(tmp_path, n_pred=n_pred, mz=False)
    for row in s.attempts.values():
        row.update(session_id="s", source_sequence="781", source_timestamp_domain="CAMERAX_SENSOR",
                   source_timestamp_nanos="112349165907000", visual_epoch="7",
                   camera_transform_generation="0", roi_policy=policy,
                   roi_left=roi[0], roi_top=roi[1], roi_right=roi[2], roi_bottom=roi[3],
                   input_width="512", input_height="512", input_scale="0.8",
                   input_pad_x="64", input_pad_y="0", mt_executed="true")
    return s


def test_historical_r0_has_full_input_not_immutable_source_frame(tmp_path):
    s = _roi_session(tmp_path)
    a = invocation_roi_audit(s, "gid")
    assert a["status"] == "FULL_MT_INPUT_CONTEXT_ONLY"
    assert a["roi_source_px"] == [0., 0., 480., 640.]
    assert a["mt_input_evidence"] is True
    assert a["full_source_frame_verified"] is False
    assert "NO_FULL_SOURCE_FRAME_EVIDENCE" in a["reasons"]
    assert a["frame_key"][1] == "781"


def test_roi_partial_does_not_call_out_of_roi_plate_a_false_negative(tmp_path):
    s = _roi_session(tmp_path, policy="r1_tracked_roi", roi=(100, 60, 400, 380))
    audit = invocation_roi_audit(s, "gid")
    assert audit["status"] == "PARTIAL_OR_UNKNOWN_MT_ROI_CONTEXT_ONLY"
    set_count(s, "gid", 5)
    assert formal_multi_plate_gate(s, "gid", s.review.invocation_annotations["gid"]["multi_plate_v1"])["fn"] is None
    assert "FORMAL_FULL_FRAME_NOT_VERIFIED" in formal_multi_plate_gate(
        s, "gid", s.review.invocation_annotations["gid"]["multi_plate_v1"])["reasons"]


def test_multiple_detection_rows_must_agree_on_same_roi(tmp_path):
    s = _roi_session(tmp_path, n_pred=2)
    s.attempts["at2"]["roi_right"] = "320"
    audit = invocation_roi_audit(s, "gid")
    assert audit["status"] == "ROI_UNVERIFIED"
    assert "INCONSISTENT_INVOCATION_GEOMETRY_OR_FRAME" in audit["reasons"]


def test_same_invocation_must_agree_on_timestamp(tmp_path):
    s = _roi_session(tmp_path, n_pred=2)
    s.attempts["at2"]["source_timestamp_nanos"] = "112349165907001"
    assert "INCONSISTENT_INVOCATION_GEOMETRY_OR_FRAME" in invocation_roi_audit(s, "gid")["reasons"]


def test_invalid_roi_and_letterbox_do_not_yield_proven_geometry(tmp_path):
    s = _roi_session(tmp_path, roi=(500, 0, 480, 640))
    a = invocation_roi_audit(s, "gid")
    assert a["roi_source_px"] is None
    assert "ROI_GEOMETRY_MISSING_OR_INVALID" in a["reasons"]
    s.attempts["at1"].update(roi_right="640", input_scale="nan")
    a = invocation_roi_audit(s, "gid")
    assert "INPUT_TRANSFORM_UNVERIFIED" in a["reasons"]


def test_r0_roi_with_nonzero_origin_fails_policy_claim(tmp_path):
    s = _roi_session(tmp_path, roi=(30, 0, 480, 640))
    a = invocation_roi_audit(s, "gid")
    assert a["status"] == "ROI_UNVERIFIED"
    assert "R0_ROI_ORIGIN_CONFLICT" in a["reasons"]


def test_missing_input_jpeg_is_explicit_not_inferred_from_crop(tmp_path):
    s = _roi_session(tmp_path)
    s.entry_names.clear()
    a = invocation_roi_audit(s, "gid")
    assert a["mt_input_evidence"] is False
    assert "MT_INPUT_EVIDENCE_MISSING" in a["reasons"]


def test_forged_source_frame_path_does_not_waive_sha_and_orientation(tmp_path):
    s = _roi_session(tmp_path)
    s.attempts["at1"].update(source_frame_entry="samples/source_frames/image.jpg",
                             source_frame_sha256="a"*64, source_frame_width=480, source_frame_height=640)
    s.entry_names.add("samples/source_frames/image.jpg")
    audit = invocation_roi_audit(s, "gid")
    assert audit["full_source_frame_verified"] is False
    assert "SOURCE_FRAME_V2_NOT_VALIDATED" in audit["reasons"]


def test_formal_gate_denies_even_if_review_mode_claims_blinded(tmp_path):
    s = _roi_session(tmp_path)
    inv = {"schema": "alpr.mobile.mt_multi_plate.v1", "plates": [
        {"plate_id": "GT01", "gt_review_mode": "blinded_independent_frozen_v1",
         "polygon_source_frame_px": [[1,1],[2,1],[2,2],[1,2]]}],
         "frozen_matching_protocol": {"iou": 0.5}}
    s.review.review_mode = "blinded_manual_v1"
    gate = formal_multi_plate_gate(s, "gid", inv)
    assert gate["status"] == "NOT_EVALUABLE"
    assert "LEGACY_MULTI_PLATE_V1_OPERATOR_MAPPING_ONLY" in gate["reasons"]
    assert (gate["tp"],gate["fp"],gate["fn"]) == (None,None,None)


def test_five_plates_diagnostic_kept_but_formal_numbers_are_null(tmp_path):
    s = populated(MultiSession(tmp_path, n_pred=4, mz=False))
    report = multi_scientific_summary(s)
    assert report["status"] == "OPERATOR_VERIFIED"
    assert report["verified_multi_invocations"] == 1
    assert report["operator_diagnostic"] == {
        "plate_count":5, "linked_predictions":4,
        "unmatched_plate_annotations":1,"marked_not_plate":0}
    assert (report["tp"],report["fp"],report["fn"],report["gt_plates"]) == (None,None,None,None)
    assert report["formal_status"] == "NOT_EVALUABLE"
    assert report["formal_issues"][0]["status"] == "NOT_EVALUABLE"
    assert report["rows"][-1]["mt_outcome"] is None
    assert report["rows"][-1]["operator_mapping"] == "no_mt_link"
    assert json.loads(json.dumps(report))["fn"] is None


def test_no_detection_is_still_executed_not_model_not_run(tmp_path):
    s = _roi_session(tmp_path, n_pred=0)
    # Synthetic invocation without detections still has one source record.
    record = dict(id="none",mt_status="NO_DETECTION", mt_detection_count="0",
                  session_id="s", source_sequence="20", source_timestamp_domain="CAMERAX_SENSOR",
                  source_timestamp_nanos="200", visual_epoch="2", camera_transform_generation="0",
                  roi_policy="r0_full_frame", roi_left="0", roi_top="0", roi_right="480", roi_bottom="640",
                  input_width="512", input_height="512", input_scale="0.8", input_pad_x="64",
                  input_pad_y="0", mt_executed="true")
    s.mt_invocations["gid"] = SimpleNamespace(
        executed=True, cancelled=False, execution_failed=False,
        records=(record,),evidence_entries=("evidence.jpg",))
    audit = invocation_roi_audit(s, "gid")
    assert audit["status"] == "FULL_MT_INPUT_CONTEXT_ONLY"
    assert formal_multi_plate_gate(s, "gid", None)["fn"] is None


def test_report_text_does_not_publish_legacy_operator_tp_fn():
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] /
              "auto_annotation_tool/ranking/mobile_scientific_review.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "Formalne TP / FP / FN: NIEOBLICZALNE" in source
    assert "przypisania operatora" in source.lower()
    assert "Tablice GT: {multi['gt_plates']} • TP:" not in source


def test_actual_markdown_never_mislabels_operator_links_as_tp(tmp_path):
    from auto_annotation_tool.ranking.mobile_scientific_review import scientific_markdown
    s = populated(MultiSession(tmp_path, n_pred=4, mz=False))
    multi = multi_scientific_summary(s)
    report = dict(scientific_state="PRELIMINARY", session_id="s",
                  source_archive_sha256="a"*64, review_revision=1,
                  review_mode="assisted", review_status="IN_PROGRESS",
                  conditions=dict(scenario="TV", variant="B", device="A12", android="Android",
                                  models={"MT":{"name":"MTn","precision":"FP32"},
                                          "MZ":{"name":"MZs","precision":"FP32"}}),
                  progress=dict(mt_invocations_cardinality_reviewed=1, mt_executed_active=1,
                                subjects_mt_done=1, subjects_total=1),
                  mt=dict(localized_n=1, single_plate_n=1, failed_localization_n=0,
                          no_detection_n=0,false_detections_confirmed=0),
                  pose=dict(polygon_iou_mean_pct=None, nmce_mean_pct=None,
                            polygon_iou_median_pct=None,nmce_median_pct=None,
                            measured_n=0,gt_missed_plate_n=1,manual_assisted_gt_n=0,
                            manual_assisted_iou_mean_pct=None,manual_assisted_nmce_mean_pct=None,
                            assisted_corrections=dict(assisted_correction_n=0,
                                                      assisted_delta_iou_mean_pct=None,
                                                      assisted_delta_nmce_mean_pct=None)),
                  mz=dict(exact_n=0,evaluable_reads=0,cer=None,gt_characters=0,error_characters=0),
                  alpr=dict(exact_n=0,evaluable_subjects=0),
                  performance=dict(latency={k:dict(n=0,p50_ms=None,p95_ms=None)
                                            for k in ('mt','mz','pipeline')},
                                   frame_flow=dict(received=None,processed=None,skipped_frame_gate=None)),
                  methods=[], caveats=[], multi_plate=multi)
    text=scientific_markdown(report)
    assert "Formalne TP / FP / FN: NIEOBLICZALNE" in text
    assert "4 powiązań MT" in text
    assert "TP: 4" not in text and "FN: 1" not in text
    assert "MZ: brak zakończonych porównań" in text
