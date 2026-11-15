"""Z4MOBREV032: NMS truth from trace, never fake GT and never count raw suppressed as FP."""
from types import SimpleNamespace
from pathlib import Path
import json
import hashlib
import zipfile
import pytest
from auto_annotation_tool.ranking.mobile_mt_invocations import group_mt_invocations, calculate_mt_invocations
from auto_annotation_tool.ranking.mobile_mt_postdedup import post_nms_proof,eligible_suppressed_id,suppressed_subject,bbox_iou
from auto_annotation_tool.ranking.mobile_verification_policy import subject_verification_issues


def make_session(tmp_path, *, raw=2, after=1, suppressed=1, mz_runs=1,
                 one_plate=False, frame_has_evidence=True, fail_iou=False):
    sid="S1"
    base={"session_id":sid,"scene_generation":161,"source_sequence":72783,"source_timestamp_nanos":551687560731000,
          "roi_policy":"r0_full_frame","mt_invocation_id":"S1-a00019","mt_detection_count":2,"mt_executed":True,
          "mt_status":"VALID_QUAD","input_width":512,"input_height":512,"plate_left":364.33182,
          "plate_top":412.5946,"plate_right":557.73065,"plate_bottom":466.03772,
          "mt_input_evidence_entry":"samples/evidence/test.jpg"}
    a=dict(base,attempt_id="A",id="A",subject_key="subA",mt_detection_index=0,
           mz_status="READ",plate_confidence=.366)
    b=dict(base,attempt_id="B",id="B",subject_key="subB",mt_detection_index=1,
           mz_status="NOT_RUN",plate_confidence=.279,plate_left=364.51294 if not fail_iou else 5,
           plate_top=413.317,plate_right=555.4981 if not fail_iou else 50,
           plate_bottom=466.07883)
    rows={v["id"]:v for v in (a,b)}
    gs=group_mt_invocations(rows,sid,strict_contract=True)
    trace={"source_sequence":72783,"source_timestamp_ns":551687560731000,
           "scene_generation":161,"frame_id":2328,
           "counters":{"plate_detections_raw":raw,"plate_detections_after_dedup":after,
                       "plate_detections_suppressed":suppressed,"mt_runs_this_frame":1,"mz_runs":mz_runs}}
    p=tmp_path/"source.alprsession"
    with zipfile.ZipFile(p,"w") as z:
        z.writestr("telemetry/traces.jsonl",json.dumps(trace)+"\n" if frame_has_evidence else "")
    annotations={"A":{"is_plate":True,"pose_quality":"accurate","evaluable":True},
                 "B":{"is_plate":True,"pose_quality":"accurate","evaluable":True}}
    review=SimpleNamespace(attempt_annotations=annotations,subjects={},sample_annotations={},
                           invocation_annotations={},
                           mt_review_policy="invocation_v1",review_status="IN_PROGRESS")
    s=SimpleNamespace(path=p,bundle=SimpleNamespace(source_archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
            mt_invocations=gs,attempts=rows,attempts_available=True,sample_ids_by_attempt={},
            invocation_by_attempt={"A":"S1-a00019","B":"S1-a00019"},
            entry_names={"samples/evidence/test.jpg"},review=review,
            subjects={"subA":{"attempts":["A"],"samples":[]},
                      "subB":{"attempts":["B"],"samples":[]}},samples={})
    s.invocation_annotation=lambda gid: {"visible_plate_count":"one" if one_plate else "multiple","evaluable":True}
    s.cancelled=lambda row:False
    s.predictions_visible=lambda key:True
    return s


def test_same_mt_call_two_raw_boxes_but_one_final_detection(tmp_path):
    s=make_session(tmp_path,one_plate=True)
    proof=post_nms_proof(s,"S1-a00019")
    assert proof and proof["raw"]==2 and proof["after_dedup"]==1 and proof["suppressed"]==1
    assert proof["trace_frame_id"]==2328 and proof["source_sequence"]==72783
    assert proof["bbox_iou"]==pytest.approx(.97343180425,rel=1e-8)
    assert eligible_suppressed_id(s,"S1-a00019")=="B"
    assert suppressed_subject(s,s.subjects["subB"])
    counts,details=calculate_mt_invocations(s)
    assert counts["evaluable_mt_invocations"]==1
    assert counts["mt_successful_invocations"]==1
    assert counts["mt_false_detections"]==0
    assert counts["mt_raw_suppressed_candidates"]==1
    assert details[0]["raw_detection_count"]==2
    assert details[0]["post_nms_detection_count"]==1
    assert not subject_verification_issues(s,"subB")


def test_raw_iou_without_trace_or_operator_not_classified(tmp_path):
    s=make_session(tmp_path,one_plate=False)
    assert post_nms_proof(s,"S1-a00019") is not None
    assert not eligible_suppressed_id(s,"S1-a00019")
    assert not suppressed_subject(s,s.subjects["subB"])
    assert calculate_mt_invocations(s)[0]["mt_raw_suppressed_candidates"]==0
    s=make_session(tmp_path,one_plate=True,after=2,suppressed=0)
    assert post_nms_proof(s,"S1-a00019") is None
    assert not suppressed_subject(s,s.subjects["subB"])
    s=make_session(tmp_path,one_plate=True,frame_has_evidence=False)
    assert post_nms_proof(s,"S1-a00019") is None


def test_reject_wrong_cardinality_or_extra_ocr(tmp_path):
    s=make_session(tmp_path,one_plate=True,fail_iou=True)
    assert not post_nms_proof(s,"S1-a00019")
    s=make_session(tmp_path,one_plate=True)
    s.attempts["B"]["mz_status"]="READ"
    assert post_nms_proof(s,"S1-a00019") is None


def test_canvas_review_is_never_auto_suppressed(tmp_path):
    s=make_session(tmp_path,one_plate=True)
    s.review.invocation_annotations["S1-a00019"]={"canvas_review_v1":{"plates":[{"gt_text":"DZA7WK7"}]}}
    original=s.invocation_annotation
    s.invocation_annotation=lambda gid: dict(original(gid),canvas_review_v1={"plates": [{"gt_text":"DZA7WK7"}]})
    assert post_nms_proof(s,"S1-a00019") is not None
    assert not eligible_suppressed_id(s,"S1-a00019")
    assert not suppressed_subject(s,s.subjects["subB"])
    # The already-existing canvas evaluation keeps its own denominator.
    assert calculate_mt_invocations(s)[0]["mt_raw_suppressed_candidates"]==0


def test_raw_suppressed_must_never_be_false_positive_or_new_gt(tmp_path):
    s=make_session(tmp_path,one_plate=True)
    # Even a legacy wrong "not a plate" decision must not turn a removed
    # pre-NMS box into a false positive of the *deployed* final detector.
    s.review.attempt_annotations["B"]={"is_plate":False,"evaluable":False}
    assert calculate_mt_invocations(s)[0]["mt_false_detections"]==0
    assert suppressed_subject(s,s.subjects["subB"])
