"""Z4MOBREV007 compatibility: manual quality grade superseded by numeric MT GT.

Older `pose_quality` decisions remain intact for audit but no longer satisfy GT
geometry requirements or appear in numeric metrics.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from auto_annotation_tool.gui import z4_mobile_sample_review as ui
from auto_annotation_tool.ranking.mobile_pose_metrics import POSE_GT_SCHEMA


class FakeSession:
    def __init__(self):
        self.session_id = "s"
        self.review = SimpleNamespace(source_archive_sha256="x"*64, attempt_annotations={
            "a1": {"is_plate": True, "evaluable": True, "pose_quality": "accurate"},
            "a2": {"is_plate": False, "evaluable": False},
        })
        self.attempts = {}
        self.entry_names = {"samples/evidence/a1.jpg", "samples/evidence/a2.jpg"}
        self.samples = {}
        self.sample_ids_by_attempt = {}
        self.mt_invocations = {}
        self.calls = []
        for i in (1, 2):
            aid, cid = f"a{i}", f"c{i}"
            entry = f"samples/evidence/{aid}.jpg"
            self.attempts[aid] = dict(id=aid, mt_invocation_id=f"i{i}", mt_status="VALID_QUAD",
                input_width=100, input_height=100, input_scale=1, input_pad_x=0,
                input_pad_y=0, roi_left=0, roi_top=0, roi_right=100, roi_bottom=100,
                plate_left=10, plate_top=10, plate_right=40, plate_bottom=20,
                mt_input_evidence_entry=entry)
            self.samples[cid] = dict(attempt_id=aid, image_entry=f"samples/crops/{cid}.jpg",
                plate_geometry={"available": True, "source_width_px":100,"source_height_px":100,
                  "plate_corners_norm":[(0.1,0.1),(0.4,0.1),(0.4,0.2),(0.1,0.2)]})
            self.sample_ids_by_attempt[aid]=[cid]
            self.mt_invocations[f"i{i}"]=SimpleNamespace(
                evidence_entries=(entry,), records=(self.attempts[aid],),
                executed=True, cancelled=False, execution_failed=False)
    def cancelled(self, row):
        return False
    def annotate(self, kind, key, **kwargs):
        self.calls.append((kind, key, kwargs))
        self.review.attempt_annotations.setdefault(key, {}).update(kwargs)


def make_view(session):
    view = ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    view.session = session
    view.current_invocation = "i1"
    view.current_record = ("sample", "c1")
    view._image_entry = "samples/evidence/a1.jpg"
    view._image_source_size = (100,100)
    view._pose_gt_cache_token = None
    view._pose_gt_points = []
    view._saving = False
    return view


def test_old_pose_quality_does_not_fabricate_a_numeric_result():
    result = ui._pose_geometry_rows(FakeSession())
    assert result[0]["metric_status"] == "missing_manual_gt"
    assert result[0]["nmce_pct"] is None
    assert result[0]["polygon_iou"] is None


def test_mismatched_evidence_is_not_scored_as_gt():
    session = FakeSession()
    session.review.attempt_annotations["a1"].update(
        pose_gt_schema=POSE_GT_SCHEMA,
        pose_gt_image_entry="samples/evidence/other.jpg",
        pose_gt_input_size_px=[100.0,100.0],
        pose_gt_quad_mt_input_px=[[10,10],[40,10],[40,20],[10,20]])
    assert ui._pose_geometry_rows(session)[0]["metric_status"] == "gt_provenance_mismatch"


def test_valid_gt_on_matching_mt_input_is_measured():
    session = FakeSession()
    session.review.attempt_annotations["a1"].update(
        pose_gt_schema=POSE_GT_SCHEMA,
        pose_gt_image_entry="samples/evidence/a1.jpg",
        pose_gt_input_size_px=[100.0,100.0],
        pose_gt_quad_mt_input_px=[[10,10],[40,10],[40,20],[10,20]])
    row=ui._pose_geometry_rows(session)[0]
    assert row["metric_status"] == "measured"
    assert row["nmce_pct"] == 0 and row["polygon_iou"] == 1


def test_false_detection_does_not_enter_polygon_metric_denominator():
    summary = ui._pose_geometry_summary(FakeSession())
    assert summary["pose_eligible"] == 1
    assert summary["pose_measured"] == 0
    assert summary["pose_pending"] == 1


def test_gt_context_requires_same_attempt_and_exact_input_image():
    session = FakeSession()
    view = make_view(session)
    assert view._pose_gt_attempt_context() == "a1"
    view._image_entry = "samples/crops/c1.jpg"
    assert view._pose_gt_attempt_context() is None
    view._image_entry = "samples/evidence/a1.jpg"
    view.current_invocation = "i2"
    assert view._pose_gt_attempt_context() is None


def test_no_subjective_pose_radios_and_no_gt_approval_cta():
    source = inspect.getsource(ui.MobileSampleReviewWindow._build)
    assert "pose_quality_radios" not in source
    assert "accept_button" not in source
    assert "Zapisz prawidłowy numer" not in source
    assert "pose_edit_button" in source
    assert "pose_edit_checkbox" not in source


def test_reopening_loads_manual_gt_from_sidecar_only():
    session = FakeSession()
    saved = [[11,12],[42,12],[42,23],[11,23]]
    session.review.attempt_annotations["a1"].update(
        pose_gt_schema=POSE_GT_SCHEMA,pose_gt_image_entry="samples/evidence/a1.jpg",
        pose_gt_input_size_px=[100.0,100.0],pose_gt_quad_mt_input_px=saved)
    view=make_view(session)
    view._ensure_pose_gt_points("a1")
    assert view._pose_gt_points == [tuple(p) for p in saved]
    view._ensure_pose_gt_points("a1")
    assert view._pose_gt_points == [tuple(p) for p in saved]


def test_missing_source_mt_input_does_not_block_review_or_create_metric():
    session = FakeSession()
    session.entry_names.remove("samples/evidence/a1.jpg")
    row=ui._pose_geometry_rows(session)[0]
    assert row["metric_status"] == "no_mt_input_evidence"
    assert ui._pose_geometry_summary(session)["pose_eligible"] == 0
