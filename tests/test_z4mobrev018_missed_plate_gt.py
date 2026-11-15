"""Z4MOBREV018: manual GT for actual MT NO_DETECTION, without fake Pose.

No camera access, no GUI display, no archive or sidecar mutation.
"""
from types import SimpleNamespace
import pytest
from PIL import Image
from auto_annotation_tool.gui import z4_mobile_sample_review as mod


ENTRY = "samples/evidence/attempt-5.jpg"
GT = [[150.0, 160.0], [260.0, 155.0], [265.0, 183.0], [151.0, 187.0]]


class Slot:
    def __init__(self, value=None):
        self.value=value
        self.calls=[]
    def set(self, value):
        self.value=value
    def get(self):
        return self.value
    def configure(self, **kwargs):
        self.calls.append(kwargs)
    def grid(self, **kwargs):
        self.calls.append(("grid",kwargs))
    def grid_remove(self):
        self.calls.append(("hide",))


class Canvas:
    def __init__(self): self.calls=[]
    def create_polygon(self,*coords,**kwargs):
        self.calls.append(("polygon",coords,kwargs))
    def create_line(self,*coords,**kwargs):
        self.calls.append(("line",coords,kwargs))
    def create_oval(self,*coords,**kwargs):
        self.calls.append(("oval",coords,kwargs))
    def create_text(self,*coords,**kwargs):
        self.calls.append(("text",coords,kwargs))
    def configure(self,**kwargs):
        self.calls.append(("configure",(),kwargs))


def new_view(*, count="one", status="NO_DETECTION", cancelled=False,
             detection_count=0, two_records=False, image_entry=ENTRY,
             source_size=(512.0,512.0)):
    attempt = {"id":"attempt-5", "mt_status":status,"mt_invocation_id":"inv-5",
               "input_width":"512","input_height":"512", "mt_input_evidence_entry":ENTRY,
               "stale_or_cancelled": "true" if cancelled else "false"}
    records=[attempt] + ([dict(attempt,id="other")] if two_records else [])
    group=SimpleNamespace(records=records, detection_count=detection_count,
          cancelled=False, executed=True, execution_failed=False, evidence_entries=(ENTRY,))
    data={"visible_plate_count":count,"evaluable":count != "uncertain"}
    sess=SimpleNamespace(
        attempts={"attempt-5":attempt}, samples={}, sample_ids_by_attempt={},
        mt_invocations={"inv-5":group}, entry_names={ENTRY}, session_id="session-018",
        review=SimpleNamespace(attempt_annotations={},source_archive_sha256="sha-018"),
        invocation_annotation=lambda _key:data,
        cancelled=lambda row: row.get("stale_or_cancelled") == "true")
    def annotate(kind,key,**decision):
        assert kind == "attempt" and key == "attempt-5"
        sess.review.attempt_annotations[key] = dict(sess.review.attempt_annotations.get(key, {}), **decision)
    sess.annotate=annotate
    view=mod.MobileSampleReviewWindow.__new__(mod.MobileSampleReviewWindow)
    view.current_record=("attempt","attempt-5")
    view.current_invocation="inv-5"
    view.subject_key="subject-5"
    view.session=sess
    view._saving=False
    view._image_entry=image_entry
    view._image_source_size=source_size
    view._image=Image.new('RGB',(512,512),'black')
    view._view_center=(256.0,256.0)
    view._view_pixel_scale=1.0
    view._pose_gt_cache_token=None
    view._pose_gt_points=[]
    view._pose_gt_drag_index=None
    view._pose_gt_before_drag=None
    view._view_drag_at=None
    view.pose_edit_var=Slot(False)
    view.canvas=Canvas()
    view.status_var=Slot("")
    view._paint_image=lambda *a,**kw:None
    view._sync_pose_gt_controls=lambda:None
    view._change=lambda operation,**kw:operation()
    return view,sess


def test_one_operator_confirmed_missed_plate_can_be_annotated_without_model_pose():
    view,_=new_view()
    assert view._pose_attempt_context() is None
    assert view._missed_plate_attempt_context() == "attempt-5"
    assert view._pose_gt_attempt_context() == "attempt-5"
    assert view._gt_target_is_confirmed("attempt-5")


@pytest.mark.parametrize('options',[{'count':'none'}, {'count':'multiple'}, {'count':'uncertain'},
    {'cancelled':True}, {'two_records':True}, {'detection_count':1},
    {'image_entry':'other.jpg'}, {'source_size':(320.,320.)}])
def test_not_eligible_when_not_exactly_one_live_missed_plate(options):
    view,_=new_view(**options)
    assert view._missed_plate_attempt_context() is None
    assert view._pose_gt_attempt_context() is None


def test_writes_independent_gt_without_promoting_a_missing_detection():
    view,sess=new_view()
    view._pose_gt_points=[tuple(x) for x in GT]
    view._pose_gt_drag_index=3
    view.pose_edit_var.set(True)
    view._finish_pose_gt_drag()
    data=sess.review.attempt_annotations['attempt-5']
    assert data['pose_gt_quad_mt_input_px']==GT
    assert data['pose_gt_schema']==mod.POSE_GT_SCHEMA
    assert data['pose_gt_image_entry']==ENTRY
    assert data['pose_gt_input_size_px']==[512.,512.]
    assert data['pose_gt_review_mode']=='assisted_gt_missed_mt_detection_v1'
    assert 'is_plate' not in data and 'plate_visibility' not in data and 'evaluable' not in data
    assert sess.attempts['attempt-5']['mt_status']=='NO_DETECTION'
    assert not view.pose_edit_var.get()


def test_unpredicted_gt_is_preserved_in_report_without_iou_or_nmce():
    view,sess=new_view()
    view._pose_gt_points=[tuple(x) for x in GT]
    view._pose_gt_drag_index=3
    view._finish_pose_gt_drag()
    rows=mod._pose_geometry_rows(sess)
    assert len(rows)==1
    assert rows[0]['metric_status']=='missed_detection_gt_saved'
    assert rows[0]['gt_quad_mt_input_px']!=''
    assert rows[0]['pred_quad_mt_input_px']==''
    assert rows[0]['eligible'] is False
    assert rows[0]['nmce_pct'] is None and rows[0]['polygon_iou'] is None
    summary=mod._pose_geometry_summary(sess)
    assert summary['pose_measured']==0 and summary['pose_eligible']==0
    assert summary['missed_detection_manual_gt']==1


def test_conflicting_plate_count_marks_stored_gt_without_comparison():
    view,sess=new_view()
    view._pose_gt_points=[tuple(x) for x in GT]
    view._pose_gt_drag_index=3
    view._finish_pose_gt_drag()
    old=sess.invocation_annotation
    sess.invocation_annotation=lambda _key:{'visible_plate_count':'none','evaluable':True}
    rows=mod._pose_geometry_rows(sess)
    assert rows[0]['metric_status']=='missed_detection_gt_cardinality_conflict'
    assert rows[0]['polygon_iou'] is None and rows[0]['nmce_pct'] is None
    assert mod._pose_geometry_summary(sess)['missed_detection_manual_gt']==0
    sess.invocation_annotation=old


def test_bad_gt_geometry_not_counted_even_when_status_valid():
    view,sess=new_view()
    sess.review.attempt_annotations['attempt-5']={
        'pose_gt_schema':mod.POSE_GT_SCHEMA,
        'pose_gt_image_entry':ENTRY,
        'pose_gt_input_size_px':[512.,512.],
        'pose_gt_quad_mt_input_px':[[250,250]]*4}
    rows=mod._pose_geometry_rows(sess)
    assert rows[0]['metric_status']=='missed_detection_gt_invalid'
    assert rows[0]['nmce_pct'] is None


def test_no_model_polygon_or_bbox_is_fabricated_on_miss():
    view,sess=new_view()
    view._pose_gt_points=[tuple(x) for x in GT]
    view._pose_gt_drag_index=3
    view._finish_pose_gt_drag()
    # The viewer also restores manual GT from the sidecar (never from MT BBOX).
    view._pose_gt_cache_token=None
    view._pose_gt_points=[]
    view._paint_detection_overlay(512,512,0,0)
    kinds=[item[0] for item in view.canvas.calls]
    tags=[item[2].get('tags',()) for item in view.canvas.calls]
    assert 'pose_gt_polygon' in {v for item in tags for v in item}
    assert 'detection_polygon' not in {v for item in tags for v in item}
    assert 'detection_box' not in {v for item in tags for v in item}
    assert 'rectangle' not in kinds
    assert len([c for c in view.canvas.calls if c[0]=='oval'])==4


def test_wrong_image_never_displays_gt_overlay():
    view,sess=new_view()
    view._pose_gt_points=[tuple(x) for x in GT]
    view._pose_gt_drag_index=3
    view._finish_pose_gt_drag()
    view._image_entry='samples/evidence/other.jpg'
    view.canvas.calls.clear()
    view._paint_detection_overlay(512,512,0,0)
    assert not any(x[0]=='polygon' for x in view.canvas.calls)


def test_no_detection_without_manual_gt_has_no_pose_metric():
    _,sess=new_view()
    row=mod._pose_geometry_rows(sess)[0]
    assert row['metric_status']=='no_valid_pose'
    assert row['nmce_pct'] is None and row['polygon_iou_pct'] is None


def test_valid_mt_detection_without_pose_remains_available_for_manual_gt():
    view,sess=new_view(status='VALID_QUAD',detection_count=1)
    assert view._pose_attempt_context()=='attempt-5'
    assert view._pose_gt_attempt_context()=='attempt-5'
    assert not view._gt_target_is_confirmed('attempt-5')
    sess.review.attempt_annotations['attempt-5']={'is_plate':True,'evaluable':True}
    assert view._gt_target_is_confirmed('attempt-5')
