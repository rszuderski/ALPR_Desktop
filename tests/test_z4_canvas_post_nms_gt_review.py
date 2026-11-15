"""AKT61: canvas post-NMS RAW is audit evidence, never a second GT/FP.

Synthetic sessions; the actual Android trace parser is separately tested in
`test_z4mobrev032_trace_proven_post_nms.py`. Here the verified helper result is
controlled, so the tests exercise Canvas population and scoring deterministically.
"""
import copy
from types import SimpleNamespace

import pytest
from auto_annotation_tool.ranking import mobile_canvas_review as core


def rectangle(x1, y1, x2, y2):
    return [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]


class FakeSession:
    def __init__(self):
        self.review = SimpleNamespace(source_archive_sha256="sha256-source", invocation_annotations={},
            attempt_annotations={}, subjects={}, review_revision=0, review_status="IN_PROGRESS")
        self.records = [
            {"id":"A", "mt_status":"VALID_QUAD", "mt_detection_index":0,
             "quad":rectangle(20, 20, 120, 60)},
            {"id":"B", "mt_status":"VALID_QUAD", "mt_detection_index":1,
             "quad":rectangle(22, 21, 121, 61)}]
        self.mt_invocations={"G":SimpleNamespace(records=self.records,executed=True,cancelled=False,
            execution_failed=False,detection_count=2)}
        self.attempts={"A":{"mz_status":"READ", "raw_prediction":"ABC123"},
                       "B":{"mz_status":"NOT_RUN"}}
        self.samples={"sA":{"raw_prediction":"ABC123"}}
        self.sample_ids_by_attempt={"A":("sA",),"B":()}
        self.subjects={}
    def cancelled(self, row):
        return False
    def invocation_annotation(self, gid):
        return self.review.invocation_annotations.get(gid,{})
    def _save_change(self, review):
        review.review_revision = self.review.review_revision + 1
        self.review=review


@pytest.fixture
def setup(monkeypatch):
    s=FakeSession()
    proof={"primary_id":"A", "suppressed_id":"B", "mt_invocation_id":"G",
           "trace_frame_id":2328, "source_sequence":72783, "bbox_iou":.973431804,
           "raw":2, "after_dedup":1, "suppressed":1}
    state={"proof":proof}
    monkeypatch.setattr(core,"source_invocation_context",lambda rows,gid,names,image_size:dict(
        records=rows,invocation_id=gid, size=(320,240),roi=(0,0,320,240),entry="full.jpg",
        source_sequence=72783,source_timestamp_nanos=55168))
    monkeypatch.setattr(core,"source_pose_quad",lambda row,size:row["quad"])
    monkeypatch.setattr(core,"_post_nms_proof",lambda session,gid,parents:copy.deepcopy(state["proof"]))
    s.entry_names={"full.jpg"}
    return s, state


def _persist(session, s):
    core._store(session, "G", s, "synthetic_test_write")


def test_raw_two_post_nms_one_is_one_provisional_gt(setup):
    s, state=setup
    scene=core.scene(s,"G")
    assert len(scene["parents"]) == 2
    assert len(scene["objects"]) == 2
    assert scene["objects"][1]["post_nms_suppressed"] is True
    assert scene["objects"][0]["attempt_id"] == "A"
    assert scene["objects"][1]["attempt_id"] == "B"
    assert core.counts(scene)==dict(expected=1, mt=2, linked=1, missed=0,
        false_positive=0, operator_errors=0, raw_suppressed=1)
    assert core.object_review_cells(scene,scene["objects"][1])[2] == "—"
    _persist(s,scene)
    assert core.scene(s,"G")["post_nms_evidence"] == state["proof"]


def test_no_trace_proof_does_not_suppress_even_with_overlapping_boxes(setup):
    s,state=setup
    state["proof"]=None
    scene=core.scene(s,"G")
    assert core.counts(scene)["expected"]==0  # no operator GT, regardless of overlap
    assert not scene.get("post_nms_evidence")
    assert not any(o.get("post_nms_suppressed",False) for o in scene["objects"])


def test_saved_legacy_gt_not_overwritten_and_explicit_operator_resolution(setup):
    s,state=setup
    state["proof"]=None
    old=core.scene(s,"G")
    old["objects"][0].update(gt_text="ABC123",text_confirmed=True)
    old["objects"][1].update(gt_text="OTHER",text_confirmed=True,
        geometry_state="operator_corrected")
    old["checked"]=True
    _persist(s,old)
    state["proof"]={"primary_id":"A","suppressed_id":"B","mt_invocation_id":"G",
        "trace_frame_id":2328,"source_sequence":72783,"bbox_iou":.973431804,
        "raw":2,"after_dedup":1,"suppressed":1}
    pending=core.scene(s,"G")
    assert pending["_post_nms_pending"]=="B"
    assert core.counts(pending)["expected"]==2
    assert any("NMS" in i for i in core._content_issues(pending))
    assert core.summary(s)["pending_scenes"]==1
    assert core.summary(s)["expected_plates"]==0
    core.accept_post_nms_raw_only(s,"G")
    after=core.scene(s,"G")
    assert after["objects"][1]["gt_text"] == "OTHER"
    assert after["objects"][1]["geometry_state"] == "operator_corrected"
    assert after["objects"][1]["post_nms_suppressed"] is True
    assert after["checked"] is False
    assert after["events"][-1]["action"]=="operator_accept_trace_proven_nms_raw_only"
    assert after["events"][-1]["object_before"]["gt_text"] == "OTHER"
    assert core.counts(after)["expected"]==1
    assert core.counts(after)["false_positive"]==0
    core.set_scene_review_state(s,"G",checked=True)
    data=core.summary(s)
    assert data["checked_scenes"]==1
    assert data["expected_plates"]==1
    assert data["raw_suppressed_mt"]==1
    assert data["evaluated_reads"]==1
    assert data["exact_reads"]==1
    rows=data["rows"]
    assert rows[1]["outcome"]=="RAW_SUPPRESSED_POST_NMS"
    assert rows[1]["included"] is False
    assert rows[1]["gt_text"] == "OTHER"  # preserved, explicitly excluded


def test_second_real_plate_remains_in_gt_as_manual_when_mt_suppressed(setup):
    s,_=setup
    scene=core.scene(s,"G")
    scene["objects"].append(core._new_object("MANUAL:C",quad=rectangle(155,90,238,120)))
    _persist(s,scene)
    assert core.counts(core.scene(s,"G")) == dict(expected=2, mt=2, linked=1,
        missed=1, false_positive=0, operator_errors=0, raw_suppressed=1)
    assert core.scene(s,"G")["objects"][-1]["attempt_id"] == ""


def test_suppressed_raw_cannot_be_edited_as_second_gt_or_marked_fp(setup):
    s,_=setup
    scene=core.scene(s,"G")
    _persist(s,scene)
    with pytest.raises(ValueError,match="RAW"):
        core.toggle_error(s,"G","GT:B")
    with pytest.raises(ValueError,match="RAW"):
        core.remove_operator_geometry(s,"G","GT:B")
    with pytest.raises(ValueError,match="RAW"):
        core.correct_plate(s,"G","GT:B",rectangle(22,21,121,61))
    # A genuine post-NMS final false positive is still counted.
    scored=core.scene(s,"G")
    scored["objects"][0]["error"] = True
    _persist(s,scored)
    assert core.counts(core.scene(s,"G"))["false_positive"]==1
    assert core.counts(core.scene(s,"G"))["raw_suppressed"]==1


def test_missing_or_conflicting_trace_fails_closed_on_saved_nms_state(setup):
    s,state=setup
    _persist(s,core.scene(s,"G"))
    state["proof"]=None
    with pytest.raises(ValueError,match="dowodu NMS"):
        core.scene(s,"G")
    with pytest.raises(ValueError,match="dowodu NMS"):
        core._store(s,"G",s.invocation_annotation("G")[core.KEY],"incorrect_write")


def test_legacy_operator_count_without_proof_is_never_silently_reduced(setup):
    s,state=setup
    state["proof"]=None
    saved=core.scene(s,"G")
    saved["objects"][1]["error"] = True
    _persist(s,saved)
    state["proof"]={"primary_id":"A","suppressed_id":"B","mt_invocation_id":"G",
        "trace_frame_id":2328,"source_sequence":72783,"bbox_iou":.973431804,
        "raw":2,"after_dedup":1,"suppressed":1}
    unresolved=core.scene(s,"G")
    assert unresolved["objects"][1]["error"] is True
    assert unresolved["_post_nms_pending"]=="B"
    assert core.counts(unresolved)["false_positive"]==1
    core.accept_post_nms_raw_only(s,"G")
    resolved=core.scene(s,"G")
    assert resolved["objects"][1]["error"] is True  # old user decision preserved
    assert core.counts(resolved)["false_positive"]==0  # but not a final FP
    assert resolved["events"][-1]["object_before"]["error"] is True
