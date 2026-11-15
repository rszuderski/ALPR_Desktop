"""Regression and scientific gates for independent per-plate mobile MT review."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from auto_annotation_tool.ranking.mobile_multi_plate_verification import (
    SCHEMA, set_count, inventory, group_issues, assign_attempt,
    set_missed_bbox, clear_missed_bbox, set_plate_text, multi_scientific_summary,
)
from auto_annotation_tool.ranking.mobile_verification_policy import (
    visual_decision, verification_issues, verification_progress,
    observation_verification_cells,
)


class Review:
    def __init__(self):
        self.invocation_annotations={"gid":{"visible_plate_count":"multiple","evaluable":True}}
        self.attempt_annotations={}
        self.subjects={}
        self.review_status="IN_PROGRESS"
        self.review_revision=0
        self.schema="alpr.mobile.human_review.v2"
        self.review_id="r"
        self.source_archive_sha256="a"*64
        self.updated_at="2026-10-18T14:00:00Z"
        self.review_mode="assisted"
        self.provenance={}
    def to_dict(self):
        return copy.deepcopy(vars(self))


class MultiSession:
    def __init__(self,tmp_path, n_pred=4,mz=True,with_no_detection=False):
        self.session_id="s"
        self.review=Review()
        self.attempts={}
        self.samples={}
        self.sample_ids_by_attempt={}
        for i in range(n_pred):
            aid=f"at{i+1}"
            self.attempts[aid]={"id":aid,"subject_key":"s1","mt_status":"VALID_QUAD",
                "mt_detection_index":i,"mt_input_evidence_entry":"evidence.jpg",
                "input_width":640,"input_height":480,
                "mz_status":"READ" if mz else "NOT_RUN","prediction":f"ABC12{i}"}
            if mz:
                sid=f"sample{i+1}"
                self.samples[sid]={"attempt_id":aid,"mz_status":"READ","prediction":f"ABC12{i}"}
                self.sample_ids_by_attempt[aid]=[sid]
        if with_no_detection and not self.attempts:
            self.attempts["no"]={"id":"no","subject_key":"s1","mt_status":"NO_DETECTION",
                "mt_input_evidence_entry":"evidence.jpg","input_width":640,"input_height":480,"mz_status":"NOT_RUN"}
        self.subjects={"s1":{"attempts":list(self.attempts),"samples":list(self.samples)}}
        group=SimpleNamespace(executed=True,cancelled=False,execution_failed=False,
            evidence_entries=("evidence.jpg",),records=tuple(self.attempts.values()),
            detection_count=n_pred,subject_keys=frozenset({"s1"}))
        self.mt_invocations={"gid":group}
        self.invocation_by_attempt={aid:"gid" for aid in self.attempts}
        self.entry_names={"evidence.jpg"}
        self.sidecar_path=tmp_path/"human_reviews"/("a"*64+".review.json")
    def invocation_annotation(self,gid):
        return self.review.invocation_annotations.get(gid,{})
    def _save_change(self,changed):
        changed.review_revision+=1
        self.review=changed
    def cancelled(self,row):return False
    def predictions_visible(self,key):return True
    def verify_source(self,*,force=False):return True


def populated(session,n=5):
    set_count(session,"gid",n)
    for i in range(min(len(session.attempts),n)):
        aid=f"at{i+1}"
        session.review.attempt_annotations[aid]=visual_decision("accurate")
        assign_attempt(session,"gid",f"GT{i+1:02d}",aid)
        if session.attempts[aid]["mz_status"]=="READ":
            set_plate_text(session,"gid",f"GT{i+1:02d}",f"ABC12{i}")
    for i in range(len(session.attempts),n):
        set_missed_bbox(session,"gid",f"GT{i+1:02d}",(100+i,130+i,150+i,160+i),
                        "evidence.jpg",(640,480))
    return session


def test_multi_requires_exact_count(tmp_path):
    s=MultiSession(tmp_path)
    assert "dokładną liczbę" in group_issues(s,"gid")[0]
    assert verification_progress(s)["status"]=="IN_PROGRESS"
    assert not verification_progress(s)["verified"]

@pytest.mark.parametrize("bad",[0,1,33,-1,True,1.5])
def test_count_rejects_invalid_and_bounded(tmp_path,bad):
    s=MultiSession(tmp_path)
    with pytest.raises(ValueError):set_count(s,"gid",bad)
    assert inventory(s,"gid") is None


def test_five_gt_four_predictions_and_one_independent_fn(tmp_path):
    s=populated(MultiSession(tmp_path))
    assert group_issues(s,"gid")==[]
    report=multi_scientific_summary(s)
    assert report["verified_multi_invocations"]==1
    assert (report["gt_plates"],report["tp"],report["fp"],report["fn"]) == (None, None, None, None)
    assert report["formal_status"] == "NOT_EVALUABLE"
    assert report["operator_diagnostic"]["linked_predictions"] == 4
    assert report["operator_diagnostic"]["unmatched_plate_annotations"] == 1
    assert report["text_gt_reads"]==4 and report["text_exact"]==4
    assert verification_progress(s)["status"]=="READY_TO_APPROVE"
    assert observation_verification_cells(s,"s1")[-1]=="✓"


def test_missing_plate_requires_manual_bbox_not_generated_prediction(tmp_path):
    s=MultiSession(tmp_path)
    set_count(s,"gid",5)
    for i in range(4):
        s.review.attempt_annotations[f"at{i+1}"]=visual_decision("accurate")
        assign_attempt(s,"gid",f"GT{i+1:02d}",f"at{i+1}")
        set_plate_text(s,"gid",f"GT{i+1:02d}",f"GT{i+1}")
    assert any("ramki GT" in item for item in group_issues(s,"gid"))
    assert multi_scientific_summary(s)["tp"] is None  # not a scientific result
    set_missed_bbox(s,"gid","GT05",(100,100,150,130),"evidence.jpg",(640,480))
    assert group_issues(s,"gid")==[]


def test_duplicate_mapping_never_silent(tmp_path):
    s=MultiSession(tmp_path)
    set_count(s,"gid",5)
    s.review.attempt_annotations["at1"]=visual_decision("accurate")
    assign_attempt(s,"gid","GT01","at1")
    before=s.review.to_dict()
    with pytest.raises(ValueError,match="dwóch tablic"):
        assign_attempt(s,"gid","GT02","at1")
    assert s.review.to_dict()==before


def test_false_positive_cannot_be_matched(tmp_path):
    s=MultiSession(tmp_path)
    set_count(s,"gid",5)
    s.review.attempt_annotations["at1"]=visual_decision("not_plate")
    with pytest.raises(ValueError,match="prawdziwą"):
        assign_attempt(s,"gid","GT01","at1")


def test_cannot_complete_with_orphan_true_prediction(tmp_path):
    s=MultiSession(tmp_path)
    set_count(s,"gid",5)
    for i in range(4):s.review.attempt_annotations[f"at{i+1}"]=visual_decision("accurate")
    for i in range(5):set_missed_bbox(s,"gid",f"GT{i+1:02d}",(30+i,50+i,100+i,110+i),"evidence.jpg",(640,480))
    assert any("wymaga wskazania" in v for v in group_issues(s,"gid"))


def test_false_positive_extra_prediction_in_a_complete_scene(tmp_path):
    s=MultiSession(tmp_path,n_pred=5)
    set_count(s,"gid",5)
    for i in range(4):
        s.review.attempt_annotations[f"at{i+1}"]=visual_decision("accurate")
        assign_attempt(s,"gid",f"GT{i+1:02d}",f"at{i+1}")
        set_plate_text(s,"gid",f"GT{i+1:02d}",f"GT{i+1}")
    s.review.attempt_annotations["at5"]=visual_decision("not_plate")
    set_missed_bbox(s,"gid","GT05",(10,10,80,40),"evidence.jpg",(640,480))
    assert not group_issues(s,"gid")
    metric=multi_scientific_summary(s)
    assert (metric["tp"],metric["fp"],metric["fn"]) == (None, None, None)
    assert metric["operator_diagnostic"]["linked_predictions"] == 4
    assert metric["operator_diagnostic"]["marked_not_plate"] == 1


def test_uncertain_detection_must_be_explicitly_resolved(tmp_path):
    s=populated(MultiSession(tmp_path))
    s.review.attempt_annotations["at1"]=visual_decision("uncertain")
    assert group_issues(s,"gid")
    assert verification_progress(s)["status"]=="IN_PROGRESS"
    assert multi_scientific_summary(s)["verified_multi_invocations"]==0


def test_mz_text_or_exclusion_reason_required_per_matched_gt(tmp_path):
    s=populated(MultiSession(tmp_path))
    set_plate_text(s,"gid","GT01","")
    assert any("GT numeru" in q for q in group_issues(s,"gid"))
    set_plate_text(s,"gid","GT01","",exclusion_reason="nieczytelna")
    assert not group_issues(s,"gid")
    assert multi_scientific_summary(s)["text_gt_reads"]==3


def test_missing_gt_bbox_fails_with_mismatched_image_or_coordinates(tmp_path):
    s=MultiSession(tmp_path)
    set_count(s,"gid",2)
    for entry,bbox in (("evil.jpg",(10,10,25,25)),("evidence.jpg",(10,10,10,20)),
                       ("evidence.jpg",(-1,5,40,60))):
        with pytest.raises(ValueError):set_missed_bbox(s,"gid","GT01",bbox,entry,(640,480))
    assert not inventory(s,"gid")["plates"][0]["bbox_mt_px"]


def test_resize_refuses_silent_gt_deletion(tmp_path):
    s=MultiSession(tmp_path)
    set_count(s,"gid",5)
    set_missed_bbox(s,"gid","GT05",(12,12,40,40),"evidence.jpg",(640,480))
    before=s.review.to_dict()
    with pytest.raises(ValueError,match="usuń"):
        set_count(s,"gid",4)
    assert s.review.to_dict()==before
    clear_missed_bbox(s,"gid","GT05")
    set_count(s,"gid",4)
    assert len(inventory(s,"gid")["plates"])==4


def test_no_detection_with_two_gt_requires_two_manual_bboxes(tmp_path):
    s=MultiSession(tmp_path,n_pred=0,mz=False,with_no_detection=True)
    set_count(s,"gid",2)
    assert len(group_issues(s,"gid"))==2
    set_missed_bbox(s,"gid","GT01",(10,10,80,30),"evidence.jpg",(640,480))
    set_missed_bbox(s,"gid","GT02",(90,10,150,35),"evidence.jpg",(640,480))
    assert group_issues(s,"gid")==[]
    assert multi_scientific_summary(s)["fn"] is None
    assert multi_scientific_summary(s)["operator_diagnostic"]["unmatched_plate_annotations"] == 2


def test_source_review_keeps_mappings_separate_from_originals(tmp_path):
    s=populated(MultiSession(tmp_path))
    raw = json.dumps(s.review.invocation_annotations["gid"],ensure_ascii=False)
    assert SCHEMA in raw
    assert s.review.invocation_annotations["gid"]["multi_plate_v1"]["plates"][4]["attempt_id"]==""
    assert s.review.review_revision > 0


def test_python_gui_sources_compile_and_have_parent_gt_rows():
    import ast
    root=Path(__file__).resolve().parents[1]
    gui=(root/'auto_annotation_tool/gui/z4_mobile_sample_review.py').read_text(encoding='utf8')
    ast.parse(gui)
    assert 'self._multi_child_targets[cid]' in gui
    assert 'self.subject_tree.insert(iid,"end",iid=cid' in gui
    assert 'self._paint_multi_overlays(width, height)' in gui
    assert 'def _multi_save_match' in gui
    assert 'def _multi_start_bbox' in gui
    assert 'Zatwierdź weryfikację' in gui
    assert 'def _pan_end' in gui
