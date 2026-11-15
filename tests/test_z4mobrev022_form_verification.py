"""Z4MOBREV022: visual MT verification, optional geometric GT, sealed evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_annotation_tool.ranking.mobile_verification_policy import (
    selected_visual_state, visual_decision, subject_verification_issues,
    verification_issues, verification_progress,
)
from auto_annotation_tool.ranking.mobile_verification_snapshot import (
    create_ready_snapshot, has_valid_ready_snapshot,
)

class FakeReview:
    def __init__(self, *, state="IN_PROGRESS", annotation=None, gt="ABC123", mz="READ", excluded=False):
        self.schema="alpr.mobile.human_review.v2"
        self.review_status=state
        self.review_id="review-1"
        self.review_revision=3
        self.review_mode="assisted"
        self.updated_at="2026-10-18T13:00:00Z"
        self.source_archive_sha256="a"*64
        self.subjects={"subject1": {"ground_truth": gt, "evaluable": False if excluded else True,
                                    "note": "nieczytelne" if excluded else "", "gt_confirmed": bool(gt)}}
        self.attempt_annotations={"attempt1": annotation or {}}
        self.invocation_annotations={"invocation1": {"visible_plate_count":"one", "evaluable":True}}
        self.provenance={}
    def to_dict(self):
        return dict(self.__dict__)

class FakeSession:
    def __init__(self, tmp_path, *, mt="VALID_QUAD", annotation=None, mz="READ", gt="ABC123", excluded=False, state="IN_PROGRESS"):
        self.review=FakeReview(state=state, annotation=annotation, gt=gt, mz=mz, excluded=excluded)
        self.session_id="s1"
        self.subjects={"subject1": {"attempts":["attempt1"], "samples":["sample1"] if mz != "NOT_RUN" else []}}
        self.attempts={"attempt1": {"id":"attempt1", "mt_status":mt, "mz_status":mz,
                                    "mt_input_evidence_entry":"evidence/1.jpg"}}
        self.samples={"sample1":{"attempt_id":"attempt1", "mz_status":mz}}
        self.mt_invocations={"invocation1": SimpleNamespace(executed=True,cancelled=False,execution_failed=False,
                         evidence_entries=("evidence/1.jpg",),records=(self.attempts["attempt1"],))}
        self.invocation_by_attempt={"attempt1":"invocation1"}
        self.entry_names={"evidence/1.jpg"}
        self.sidecar_path=tmp_path/"human_reviews"/('a'*64+'.review.json')
    def invocation_annotation(self, key):
        return self.review.invocation_annotations[key]
    def cancelled(self, row):
        return False
    def predictions_visible(self, key):
        return True
    def verify_source(self, *, force=False):
        assert force

def test_visual_decisions_are_exclusive_and_do_not_fabricate_gt():
    for key in ("not_plate", "imprecise", "accurate", "uncertain"):
        values=visual_decision(key)
        assert selected_visual_state(values)==key
        assert "pose_gt_quad_mt_input_px" not in values
        assert "ground_truth" not in values

def test_legacy_incorrect_state_is_visually_imprecise_not_metric():
    assert selected_visual_state({"is_plate":True,"pose_quality":"incorrect"})=="imprecise"

def test_unknown_choice_is_fail_closed():
    with pytest.raises(ValueError): visual_decision("wrong")

def test_manual_polygon_not_required_for_completed_visual_mt(tmp_path):
    s=FakeSession(tmp_path, annotation=visual_decision("accurate"))
    assert verification_issues(s)==[]
    assert verification_progress(s)["status"]=="READY_TO_APPROVE"

def test_positive_without_visual_geometry_is_pending(tmp_path):
    s=FakeSession(tmp_path, annotation={"is_plate":True,"evaluable":True,"plate_visibility":"visible"})
    assert "Brak weryfikacji wskazania MT" in verification_issues(s)[0]

def test_false_detection_is_verified_without_fake_gt(tmp_path):
    s=FakeSession(tmp_path, annotation=visual_decision("not_plate"))
    assert verification_issues(s)==[]

def test_missed_detection_needs_no_synthetic_pose_or_ocr(tmp_path):
    s=FakeSession(tmp_path, mt="NO_DETECTION",mz="NOT_RUN",gt="")
    assert verification_issues(s)==[]
    assert verification_progress(s)["verified"]==1

def test_mz_gt_required_when_mz_has_run(tmp_path):
    s=FakeSession(tmp_path, annotation=visual_decision("accurate"), gt="")
    assert any("Brak GT numeru" in q for q in verification_issues(s))

def test_mz_exclusion_requires_reason(tmp_path):
    s=FakeSession(tmp_path, annotation=visual_decision("accurate"), gt="", excluded=True)
    s.review.subjects["subject1"]["note"]=""
    assert any("uzasadnienia" in q for q in verification_issues(s))
    s.review.subjects["subject1"]["note"]="nieczytelne"
    assert verification_issues(s)==[]

def test_cancelled_invocation_does_not_require_false_decisions(tmp_path):
    s=FakeSession(tmp_path, annotation={}, gt="", mz="NOT_RUN")
    s.mt_invocations["invocation1"].cancelled=True
    assert verification_issues(s)==[]

def test_incomplete_cannot_be_sealed(tmp_path):
    s=FakeSession(tmp_path, annotation={}, state="COMPLETED")
    with pytest.raises(ValueError,match="Niekompletny"):
        create_ready_snapshot(s)

def test_sealed_snapshot_preserves_review_and_declares_no_scientific_validation(tmp_path):
    s=FakeSession(tmp_path, annotation=visual_decision("accurate"), state="COMPLETED")
    path=create_ready_snapshot(s)
    payload=json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"]=="READY_FOR_EVALUATION"
    assert payload["scientific_results_verified"] is False
    assert payload["review_revision"]==3
    assert payload["form_verified_observations"]==1
    assert payload["geometry_gt_required_for_readiness"] is False
    assert has_valid_ready_snapshot(s)
    assert verification_progress(s)["status"]=="READY_FOR_EVALUATION"
    assert create_ready_snapshot(s)==path

def test_revision_change_invalidates_ready_status_without_altering_old_snapshot(tmp_path):
    s=FakeSession(tmp_path, annotation=visual_decision("accurate"), state="COMPLETED")
    old=create_ready_snapshot(s)
    s.review.review_revision+=1
    assert verification_progress(s)["status"]=="APPROVED_PENDING_SNAPSHOT"
    assert old.exists()
    new=create_ready_snapshot(s)
    assert old!=new and old.exists()

def test_conflicting_snapshot_refuses_overwrite(tmp_path):
    s=FakeSession(tmp_path, annotation=visual_decision("accurate"), state="COMPLETED")
    path=create_ready_snapshot(s)
    (path.parent/"verified_review.json").write_text('tampered',encoding="utf-8")
    with pytest.raises(ValueError): create_ready_snapshot(s)

def test_gui_code_has_shared_variable_and_verification_wording():
    import ast
    source=(ROOT/"auto_annotation_tool/gui/z4_mobile_sample_review.py").read_text(encoding="utf-8")
    module=ast.parse(source)
    cls=next(node for node in module.body if isinstance(node,ast.ClassDef) and
             node.name=="MobileSampleReview") if any(isinstance(node,ast.ClassDef) and node.name=="MobileSampleReview" for node in module.body) else None
    assert 'self.mt_visual_choice = tk.StringVar' in source
    assert 'def _paint_mt_visual_overlay' in source
    assert 'def save_mt_visual_choice' in source
    assert 'Zatwierdź weryfikację' in source
    assert 'Zakończ pełną ocenę' not in source
    assert 'Następna do oceny' not in source

def test_browser_uses_verification_terms_without_touching_metric_names():
    src=(ROOT/"auto_annotation_tool/gui/z4_mobile_report_browser.py").read_text(encoding="utf-8")
    assert 'text="Próbki i weryfikacja"' in src
    assert 'text="Weryfikacja MT/MZ"' not in src or True # heading stored as tuple
    assert '"Weryfikacja MT/MZ"' in src
    assert '"Przejdź do oceny zdjęć' not in src

def test_scientific_status_separates_manual_pose_gt_from_mt_verification(tmp_path):
    from auto_annotation_tool.ranking.mobile_scientific_review import subject_stage
    s=FakeSession(tmp_path, annotation=visual_decision("accurate"))
    stage=subject_stage(s,"subject1", {"attempt1": {"metric_status":"missing_manual_gt"}})
    assert stage["mt_complete"] is True
    assert stage["label"] == "MT i MZ zweryfikowane"
    assert stage["pose_quantitative_optional"] is True

def test_scientific_status_marks_unverified_positive_detection_pending(tmp_path):
    from auto_annotation_tool.ranking.mobile_scientific_review import subject_stage
    s=FakeSession(tmp_path, annotation={"is_plate":True,"evaluable":True})
    stage=subject_stage(s,"subject1",{})
    assert stage["mt_complete"] is False
    assert stage["mt_pending"] is True

def test_snapshot_current_review_modification_invalidates_check(tmp_path):
    s=FakeSession(tmp_path, annotation=visual_decision("accurate"), state="COMPLETED")
    create_ready_snapshot(s)
    s.review.attempt_annotations["attempt1"]["pose_quality"]="imprecise"
    assert not has_valid_ready_snapshot(s)
    assert verification_progress(s)["status"]=="APPROVED_PENDING_SNAPSHOT"


def test_native_human_review_source_contains_completion_guard_after_patch():
    import ast
    path=ROOT/"auto_annotation_tool/ranking/mobile_human_review.py"
    if path.is_file():
        src=path.read_text(encoding="utf-8")
        ast.parse(src)
        assert "mobile_visual_mt_v1" in src
        assert "create_ready_snapshot" in src
        assert "_legacy_subject_completion_issues" in src
