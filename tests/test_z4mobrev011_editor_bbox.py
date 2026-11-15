"""Z4MOBREV011: one-shot GT editing, auditable BBOX fallback, independent MZ exclusions."""
from __future__ import annotations
import inspect
from types import SimpleNamespace
import pytest
from auto_annotation_tool.gui import z4_mobile_sample_review as ui
from auto_annotation_tool.ranking.mobile_pose_metrics import POSE_GT_SCHEMA


class V:
    def __init__(self,value=False): self.value=value
    def set(self,value): self.value=value
    def get(self): return self.value


class C:
    def __init__(self): self.cursor=""; self.ops=[]
    def configure(self,**k): self.cursor=k.get("cursor",self.cursor)

class S:
    def __init__(self, linked=True):
        a=dict(id="a1", mt_invocation_id="inv1", mt_status="VALID_QUAD",
            mt_input_evidence_entry="frame1", input_width=100,input_height=100,
            roi_left=0,roi_top=0,roi_right=100,roi_bottom=100,
            input_scale=1,input_pad_x=0,input_pad_y=0,
            plate_left=10,plate_top=10,plate_right=40,plate_bottom=20,mt_detection_index="1")
        self.attempts={"a1":a}; self.samples={}; self.sample_ids_by_attempt={}
        if linked:
            self.samples={"c1":dict(attempt_id="a1", id="c1",plate_geometry={"available":True,
                "source_width_px":100,"source_height_px":100,
                "plate_corners_norm":[[.1,.1],[.4,.1],[.4,.2],[.1,.2]]})}
            self.sample_ids_by_attempt={"a1":["c1"]}
        self.mt_invocations={"inv1":SimpleNamespace(executed=True,cancelled=False,
            execution_failed=False,evidence_entries={"frame1"},records=[a])}
        self.entry_names={"frame1"}; self.session_id="synthetic"
        self.review=SimpleNamespace(source_archive_sha256="a"*64, subjects={"subject":dict(evaluable=False)},
            attempt_annotations={"a1":{"is_plate":True,"evaluable":True}},
            invocation_annotations={},sample_annotations={})
        self.calls=[]
    def cancelled(self,row): return False
    def annotate(self,kind,key,**k):
        self.calls.append((kind,key,k))
        self.review.attempt_annotations[key].update(k)


def view(s=None):
    s=s or S(); w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    w.session=s; w.current_record=("attempt","a1"); w.current_invocation="inv1"
    w._image_entry="frame1"; w._image_source_size=(100.,100.)
    w._image=SimpleNamespace(width=100,height=100)
    w._view_center=(50.,50.); w._view_pixel_scale=1.; w._saving=False
    w.pose_edit_var=V(False); w._pose_gt_points=[]; w._pose_gt_cache_token=None
    w._pose_gt_drag_index=None; w._pose_gt_before_drag=None; w._view_drag_at=None
    w.canvas=C(); w.status_var=V("")
    w._paint_image=lambda *args: None
    w._sync_pose_gt_controls=lambda: None
    w._change=lambda fn,**kwargs:fn()
    return w


def test_one_click_starts_one_shot_edit_and_second_click_cancels():
    w=view()
    w._toggle_pose_gt_edit()
    assert w.pose_edit_var.get() is True and w.canvas.cursor=="crosshair"
    w._pose_gt_points=[(10,10),(40,10)]
    w._pose_gt_cache_token=("a1","frame1")
    w._toggle_pose_gt_edit()
    assert w.pose_edit_var.get() is False and w._pose_gt_points==[]
    assert w.session.calls==[] and w.canvas.cursor=="hand2"


def test_four_points_autosave_end_edit_and_restore_overlay_state():
    w=view(); w.pose_edit_var.set(True)
    w._pose_gt_points=[(10,10),(40,10),(40,20),(10,20)]
    w._pose_gt_cache_token=("a1","frame1");w._pose_gt_drag_index=3
    w._pose_gt_before_drag=[]
    w._finish_pose_gt_drag()
    assert w.pose_edit_var.get() is False and w.canvas.cursor=="hand2"
    assert len(w.session.calls)==1
    _,_,data=w.session.calls[0]
    assert data["pose_gt_schema"]==POSE_GT_SCHEMA and data["pose_gt_unavailable_reason"] is None
    assert data["pose_gt_quad_mt_input_px"]==[[10,10],[40,10],[40,20],[10,20]]
    assert ui._pose_geometry_rows(w.session)[0]["metric_status"]=="measured"


def test_no_new_write_if_existing_gt_is_identical_but_edit_exits():
    w=view(); pts=[[10,10],[40,10],[40,20],[10,20]]
    w.session.review.attempt_annotations["a1"].update(
        pose_gt_schema=POSE_GT_SCHEMA,pose_gt_image_entry="frame1",pose_gt_input_size_px=[100.,100.],
        pose_gt_quad_mt_input_px=pts)
    w.pose_edit_var.set(True);w._pose_gt_points=[tuple(x) for x in pts]
    w._pose_gt_cache_token=("a1","frame1");w._pose_gt_drag_index=2;w._pose_gt_before_drag=[]
    w._finish_pose_gt_drag()
    assert not w.pose_edit_var.get() and not w.session.calls


def test_mz_text_excluded_does_not_block_mt_pose():
    s=S(); assert s.review.subjects["subject"]["evaluable"] is False
    assert ui._pose_geometry_rows(s)[0]["metric_status"]=="missing_manual_gt"
    assert ui._pose_geometry_rows(s)[0]["eligible"] is True


def test_manual_gt_visibility_exclusion_is_separate_and_auditable():
    w=view();w._set_pose_gt_unavailable()
    assert w.session.review.attempt_annotations["a1"]["pose_gt_unavailable_reason"]=="corners_not_distinguishable"
    assert w.session.review.subjects["subject"]["evaluable"] is False
    rows=ui._pose_geometry_rows(w.session)
    assert rows[0]["metric_status"]=="manual_gt_unavailable"
    assert rows[0]["eligible"] is False
    assert rows[0]["nmce_pct"] is None and rows[0]["polygon_iou"] is None
    assert ui._pose_geometry_summary(w.session)["pose_pending"]==0
    assert ui._pose_geometry_summary(w.session)["pose_manual_unavailable"]==1
    w._set_pose_gt_unavailable()
    assert w.session.review.attempt_annotations["a1"]["pose_gt_unavailable_reason"] is None
    assert ui._pose_geometry_summary(w.session)["pose_pending"]==1


def test_no_pose_when_mt_only_no_crop_but_detection_not_called_false():
    s=S(linked=False)
    key,message=ui._pose_source_status(s,s.attempts["a1"])
    assert key=="no_linked_crop" and "nie oznacza" in message
    row=ui._pose_geometry_rows(s)[0]
    assert row["pose_source_status"]=="no_linked_crop"
    assert row["metric_status"]=="no_valid_pose"
    assert row["nmce_pct"] is None and row["pred_quad_mt_input_px"]==""
    assert s.review.attempt_annotations["a1"]["is_plate"] is True


def test_crop_with_saved_corners_but_broken_transform_does_not_fabricate_quad():
    s=S();s.attempts["a1"]["input_scale"]=0
    key,message=ui._pose_source_status(s,s.attempts["a1"])
    assert key=="invalid_or_unlinked_corners"
    assert ui._pose_geometry_rows(s)[0]["polygon_iou_pct"] is None


def test_source_has_crop_but_no_saved_corners():
    s=S();s.samples["c1"]["plate_geometry"].pop("plate_corners_norm")
    assert ui._pose_source_status(s,s.attempts["a1"])[0]=="no_saved_corners"


def test_false_detection_not_changed_by_geometry_skip():
    w=view();w.session.review.attempt_annotations["a1"]["is_plate"]=False
    w._set_pose_gt_unavailable()
    assert w.session.calls==[]


def test_ux_has_no_checkbutton_and_keeps_explicit_cancel():
    source=inspect.getsource(ui.MobileSampleReviewWindow._build)
    assert "pose_edit_checkbox" not in source
    assert "pose_edit_button = ttk.Button" in source
    assert "Nie mam pewności — pomiń ocenę obrysu" in source
    assert "Pomiń ocenę odczytu numeru" in source
    save=inspect.getsource(ui.MobileSampleReviewWindow._finish_pose_gt_drag)
    assert "self.pose_edit_var.set(False)" in save


def test_annotation_provenance_conflict_is_not_scored():
    s=S();s.review.attempt_annotations["a1"].update(
        pose_gt_unavailable_reason="corners_not_distinguishable",
        pose_gt_schema=POSE_GT_SCHEMA,pose_gt_image_entry="frame1",pose_gt_input_size_px=[100.,100.],
        pose_gt_quad_mt_input_px=[[10,10],[40,10],[40,20],[10,20]])
    row=ui._pose_geometry_rows(s)[0]
    assert row["metric_status"]=="gt_exclusion_conflict"
    assert row["nmce_pct"] is None
