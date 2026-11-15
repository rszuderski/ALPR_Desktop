"""P2B2 source-coordinate semantics, data integrity and report regressions."""
from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from test_mobile_human_review import make_bundle, attempt, load
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking.mobile_verification_policy import verification_issues, observation_verification_cells
from auto_annotation_tool.ranking.mobile_package_experiments import _file_sha256
from auto_annotation_tool.ranking.mobile_pose_correction import make_correction


def source_session(tmp_path, n=2, *, mz=True, no_read=False):
    tmp_path.mkdir(parents=True,exist_ok=True)
    rows=[]; samples=[]
    for i in range(max(1,n)):
        pred="" if no_read or not mz else f"WA{i}1234"
        row=attempt(f"a{i}", prediction=pred)
        row.update(mt_invocation_id="mt-one", mt_executed=True,
            mt_detection_index=i if n else "", mt_detection_count=n,
            mt_status="VALID_QUAD" if n else "NO_DETECTION", source_sequence=17,
            source_timestamp_nanos=100000, scene_generation=1, visual_epoch=1,
            camera_transform_generation=1, source_frame_expected=True,
            source_frame_entry="samples/frames/source.jpg", source_frame_status="WRITTEN",
            source_frame_coordinate_space="mt_source_bitmap_xy",
            source_frame_origin="pipeline_bitmap_before_roi", source_frame_width_px=1000,
            source_frame_height_px=700, roi_left=100,roi_top=60,roi_right=900,roi_bottom=640,
            input_width=800,input_height=580,input_scale=1,input_pad_x=0,input_pad_y=0,
            mt_input_evidence_entry="samples/evidence/a0.jpg",
            mt_pose_status="VALID_QUAD" if n else "NO_DETECTION",
            mt_pose_coordinate_space="source_frame_normalized_v1", mt_pose_corner_order="TL_TR_BR_BL",
            mt_pose_source_width_px=1000,mt_pose_source_height_px=700,
            mt_pose_corners_norm=f"{.15+.22*i}:.2;{.30+.22*i}:.2;{.30+.22*i}:.3;{.15+.22*i}:.3")
        row["mz_status"]="READ" if mz and not no_read and n else "NO_CHARACTERS" if mz and n else "NOT_RUN"
        rows.append(row)
        if mz and n:
            samples.append(dict(capture_id=f"c{i}",attempt_id=f"a{i}",subject_key=row["subject_key"],
                prediction=pred,mz_status=row["mz_status"]))
    path=make_bundle(tmp_path/"source.alprsession",attempts=rows,samples=samples,image_size=(800,580))
    img=Image.new("RGB",(1000,700),(80,95,105));b=io.BytesIO();img.save(b,format="JPEG")
    with zipfile.ZipFile(path,"a") as z:
        z.writestr("samples/frames/source.jpg",b.getvalue())
        # This fixture represents final MT detections. Supply their source-
        # bound final/raw proof so Canvas candidate selection does not need to
        # guess FINAL from a Pose or an OCR result.
        z.writestr('telemetry/traces.jsonl',json.dumps(dict(source_sequence=17,source_timestamp_ns=100000,
            scene_generation=1,visual_epoch=1,camera_transform_generation=1,
            counters=dict(mt_runs_this_frame=1,plate_detections_raw=n,plate_detections_after_dedup=n,
                          plate_detections_suppressed=0,mz_runs=n if mz else 0))))
    session=load(path,tmp_path)
    session.set_review_mode("assisted")
    return session


def text_all(s):
    scene=core.scene(s,"mt-one")
    for obj in scene["objects"]:
        if obj["parent_id"]:
            parent=next(p for p in scene["parents"] if p["parent_id"]==obj["parent_id"])
            if parent["mz_ran"]:
                core.set_text(s,"mt-one",obj["object_id"],parent["mz_prediction"] or "WA01234")


MANUAL=[(650,350),(800,350),(800,420),(650,420)]


def test_seeding_has_two_overlapping_copies_but_not_confirmed_gt(tmp_path):
    s=source_session(tmp_path)
    before=s.review.to_dict(); initial=core.scene(s,"mt-one")
    assert initial["parents"][0]["quad"]==initial["objects"][0]["quad"]
    assert initial["objects"][0]["geometry_state"]=="unconfirmed"
    assert not initial["checked"]
    assert core.counts(initial)["expected"]==0  # two predictions, no operator GT
    assert not core.has_scene(s,"mt-one") and s.review.to_dict()==before


def test_add_four_corners_is_plus_one_and_does_not_create_mz_prediction(tmp_path):
    s=source_session(tmp_path)
    oid=core.add_plate(s,"mt-one",MANUAL)
    scene=core.scene(s,"mt-one"); obj=core._object(scene,oid)
    assert core.counts(scene)["expected"]==1  # D adds one, not the two unresolved MT copies
    assert obj["parent_id"] is None and obj["attempt_id"]==""
    row=next(r for r in core.summary(s)["rows"] if r["object_id"]==oid)
    assert row["mz_prediction"]=="" and row["mz_status"]=="NOT_RUN" and row["cer"] is None


@pytest.mark.parametrize("points",[MANUAL[:3],[(0,0),(60,0),(60,30),(0,30)],
    [(100,100),(101,101),(102,102),(103,103)],[(float('nan'),100),*MANUAL[1:]],
    [(650,350),(901,350),(901,420),(650,420)]])
def test_invalid_incomplete_or_outside_roi_never_changes_count(tmp_path,points):
    s=source_session(tmp_path);before=s.review.to_dict()
    with pytest.raises((ValueError,TypeError)):
        core.add_plate(s,"mt-one",points)
    assert s.review.to_dict()==before and core.counts(core.scene(s,"mt-one"))["expected"]==0


def test_correction_retains_original_parent_count_and_order(tmp_path):
    s=source_session(tmp_path)
    before=core.scene(s,"mt-one")
    points=copy.deepcopy(before["objects"][0]["quad"]);points[0][0]+=8
    core.correct_plate(s,"mt-one","GT:a0",points)
    after=core.scene(s,"mt-one")
    assert after["parents"]==before["parents"]
    assert after["objects"][0]["quad"]==points
    assert core.counts(after)["expected"]==1
    assert after["objects"][0]["parent_id"]=="MT:a0"


def test_bad_crossed_drag_rolls_back_whole_write(tmp_path):
    s=source_session(tmp_path);before=s.review.to_dict()
    points=core.scene(s,"mt-one")["objects"][0]["quad"]
    points[1],points[2]=points[2],points[1]
    with pytest.raises(ValueError):core.correct_plate(s,"mt-one","GT:a0",points)
    assert s.review.to_dict()==before


def test_flags_keep_geometry_and_gt_and_distinguish_model_vs_operator(tmp_path):
    s=source_session(tmp_path);text_all(s)
    manual=core.add_plate(s,"mt-one",MANUAL)
    before=core.scene(s,"mt-one")
    core.toggle_error(s,"mt-one","GT:a0")
    with pytest.raises(ValueError,match='Ręczna GT'):core.toggle_error(s,"mt-one",manual)
    core.remove_operator_geometry(s,"mt-one",manual)
    after=core.scene(s,"mt-one");c=core.counts(after)
    assert c==dict(expected=2,mt=2,linked=1,missed=1,false_positive=1,operator_errors=0,raw_suppressed=0)
    assert len(after["objects"])==3
    assert after["objects"][0]["quad"]==before["objects"][0]["quad"]
    assert after["objects"][0]["gt_text"]==before["objects"][0]["gt_text"]
    core.toggle_error(s,"mt-one","GT:a0")
    assert core.counts(core.scene(s,"mt-one"))["expected"]==2


def test_gt_is_per_plate_and_never_prediction_or_other_plate(tmp_path):
    s=source_session(tmp_path)
    core.set_text(s,"mt-one","GT:a0","wa-9999")
    sc=core.scene(s,"mt-one")
    assert sc["objects"][0]["gt_text"]=="WA9999" and not sc["objects"][1]["gt_text"]
    assert sc["parents"][0]["mz_prediction"]=="WA01234"
    assert not s.review.subjects


def test_draft_scene_never_enters_metric_denominators(tmp_path):
    s=source_session(tmp_path);text_all(s)
    d=core.summary(s)
    assert d["expected_plates"]==0 and d["evaluated_reads"]==0 and d["geometry_n"]==0
    assert d["pending_scenes"]==1 and d["polygon_iou_mean_pct"] is None


def test_ready_scene_uses_original_vs_child_iou_and_correct_gt_cer(tmp_path):
    s=source_session(tmp_path);text_all(s)
    points=core.scene(s,"mt-one")["objects"][0]["quad"]
    points[0][0]+=15
    core.correct_plate(s,"mt-one","GT:a0",points)
    core.set_text(s,"mt-one","GT:a1","WA11235")
    core.add_plate(s,"mt-one",MANUAL)
    core.check_scene(s,"mt-one")
    d=core.summary(s)
    assert (d["expected_plates"],d["linked_plates"],d["missed_plates"])==(3,2,1)
    assert d["evaluated_reads"]==2 and d["text_errors"]==1 and d["cer"]==pytest.approx(1/14)
    assert 0<d["polygon_iou_mean_pct"]<100 and d["nmce_mean_pct"]>0
    assert d["geometry_n"]==2 and not d["independent_gt"]
    assert d["mt_recall_operator"]==pytest.approx(2/3)
    assert not verification_issues(s)
    assert observation_verification_cells(s,next(iter(s.subjects)))[-1]=="✓"


def test_correcting_gt_invalidates_checked_scene(tmp_path):
    s=source_session(tmp_path);text_all(s);core.check_scene(s,"mt-one")
    core.set_text(s,"mt-one","GT:a0","ABC1234")
    assert not core.scene(s,"mt-one")["checked"] and core.summary(s)["evaluated_reads"]==0
    assert verification_issues(s)


def test_zero_detection_requires_whole_roi_confirmation(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import confirm_no_plates
    s=source_session(tmp_path,n=0,mz=False)
    assert core.counts(core.scene(s,"mt-one"))["expected"]==0
    assert not core.scene(s,"mt-one")['checked']
    core.check_scene(s,"mt-one")  # the explicit whole-image inspection suffices
    assert not verification_issues(s) and core.summary(s)["checked_scenes"]==1
    assert core.summary(s)["mt_recall_operator"] is None
    core.add_plate(s,"mt-one",MANUAL);core.check_scene(s,"mt-one")
    assert core.summary(s)["missed_plates"]==1 and core.summary(s)["mt_recall_operator"]==0
    assert core.summary(s)["cer"] is None


def test_all_predictions_false_can_be_complete_without_fabricated_gt(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import confirm_no_plates
    s=source_session(tmp_path)
    for obj in core.scene(s,"mt-one")["objects"]:core.toggle_error(s,"mt-one",obj["object_id"])
    confirm_no_plates(s,"mt-one")
    assert not verification_issues(s)
    assert core.summary(s)["expected_plates"]==0 and core.summary(s)["false_positive_mt"]==2
    assert core.summary(s)["evaluated_reads"]==0


def test_no_read_is_cer_one_but_not_run_is_not_a_fake_read(tmp_path):
    s=source_session(tmp_path/"no-read",n=1,no_read=True)
    text_all(s);core.check_scene(s,"mt-one")
    assert core.summary(s)["cer"]==1
    other=source_session(tmp_path/"not-run",n=1,mz=False)
    core.set_text(other,"mt-one","GT:a0","WA01234");core.check_scene(other,"mt-one")
    assert core.summary(other)["cer"] is None and core.summary(other)["evaluated_reads"]==0


def test_reasoned_text_exclusion_and_whole_image_exclusion(tmp_path):
    s=source_session(tmp_path)
    core.set_text(s,"mt-one","GT:a0","",reason="odbicie reflektora")
    core.set_text(s,"mt-one","GT:a1","WA11234")
    core.check_scene(s,"mt-one")
    assert core.summary(s)["evaluated_reads"]==1
    core.exclude_scene(s,"mt-one","zdjęcie rozmazane")
    assert core.summary(s)["excluded_scenes"]==1 and core.summary(s)["expected_plates"]==0
    assert not verification_issues(s)


@pytest.mark.parametrize("text,reason",[("AAA","bbb"),("A"*257,""),("","x"*241)])
def test_bad_text_write_is_atomic(tmp_path,text,reason):
    s=source_session(tmp_path);before=s.review.to_dict()
    with pytest.raises(ValueError):core.set_text(s,"mt-one","GT:a0",text,reason)
    assert s.review.to_dict()==before


def test_reopen_preserves_source_hash_parents_gt_flags_and_audit(tmp_path):
    s=source_session(tmp_path);sha=_file_sha256(s.path)
    text_all(s);manual=core.add_plate(s,"mt-one",MANUAL);core.remove_operator_geometry(s,"mt-one",manual)
    core.check_scene(s,"mt-one")
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert core.scene(reopened,"mt-one")==core.scene(s,"mt-one")
    assert _file_sha256(s.path)==sha
    assert core.scene(s,"mt-one")["events"][-1]["action"]=="confirm_complete_roi_review"


@pytest.mark.parametrize("where,change",[("source",lambda s:s.attempts["a0"].update(mt_pose_corners_norm=".16:.2;.3:.2;.3:.3;.16:.3")),
    ("parent",lambda s:s.review.invocation_annotations["mt-one"][core.KEY]["parents"][0].update(mz_prediction="FAKE")),
    ("roi",lambda s:s.review.invocation_annotations["mt-one"][core.KEY].update(roi=[0,0,1000,700])),
    ("child",lambda s:s.review.invocation_annotations["mt-one"][core.KEY]["objects"][1].update(parent_id="MT:a0")),
    ("ids",lambda s:s.review.invocation_annotations["mt-one"][core.KEY]["objects"][1].update(object_id="GT:a0"))])
def test_tamper_or_parent_reassignment_fails_closed(tmp_path,where,change):
    s=source_session(tmp_path);text_all(s);change(s)
    with pytest.raises(ValueError):core.scene(s,"mt-one")
    assert core.group_issues(s,"mt-one") and core.summary(s)["expected_plates"]==0


def test_unknown_invocation_is_safe_to_query(tmp_path):
    s=source_session(tmp_path)
    s.review.mt_review_policy="legacy_visibility_v1"
    assert not core.has_scene(s,"")
    assert not core.has_scene(s,"outside")


def test_old_count_api_cannot_erase_canvas_sidecar(tmp_path):
    s=source_session(tmp_path);text_all(s);before=s.review.to_dict()
    with pytest.raises(ValueError,match="canvas"):
        s.annotate_invocation("mt-one",visible_plate_count="one")
    assert s.review.to_dict()==before


def test_canvas_metrics_are_separate_from_legacy_subject_and_mt_denominators(tmp_path):
    s=source_session(tmp_path,n=1)
    s.set_subject(next(iter(s.subjects)),ground_truth="WA01234")
    s.annotate_invocation("mt-one",visible_plate_count="one")
    text_all(s);core.check_scene(s,"mt-one")
    stats=s.statistics()
    assert stats["summary"]["evaluable_reads"]==0 and stats["summary"]["evaluable_mt_invocations"]==0
    assert stats["canvas_review"]["evaluated_reads"]==1
    assert stats["summary"]["canvas_expected_plates"]==1


def test_export_includes_original_and_child_quads_gt_predictions_and_assisted_report(tmp_path):
    s=source_session(tmp_path);text_all(s);core.check_scene(s,"mt-one")
    paths=s.export(tmp_path/"export")
    names={p.name for p in paths}
    assert {"canvas_review_summary.json","canvas_review_plates.csv","canvas_review_report.md"}<=names
    d=json.loads((tmp_path/"export"/"canvas_review_summary.json").read_text())
    assert d["rows"][0]["original_mt_quad"] and d["rows"][0]["operator_quad"]
    assert d["rows"][0]["mz_prediction"]==d["rows"][0]["gt_text"]
    assert "asystowana" in (tmp_path/"export"/"canvas_review_report.md").read_text()


def test_legacy_verified_correction_migrates_through_proven_transform(tmp_path):
    s=source_session(tmp_path,n=1)
    source=core.scene(s,"mt-one")["parents"][0]["quad"]
    predicted=[[x-100,y-60] for x,y in source]
    changed=copy.deepcopy(predicted);changed[0][0]+=5
    c=make_correction(attempt_id="a0",source_sha=s.review.source_archive_sha256,
        image_entry="samples/evidence/a0.jpg",size=(800,580),prediction=predicted,corrected=changed)
    s.review.attempt_annotations["a0"]={"pose_mt_correction_v1":c,"is_plate":True,"pose_quality":"imprecise"}
    sc=core.scene(s,"mt-one")
    assert sc["objects"][0]["quad"][0][0]==source[0][0]+5
    assert not sc["migration_warnings"]
    assert s.review.attempt_annotations["a0"]["pose_mt_correction_v1"]==c


def test_external_sidecar_revision_conflict_never_overwrites(tmp_path):
    s=source_session(tmp_path);text_all(s)
    other=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    core.set_text(other,"mt-one","GT:a0","BBBB")
    saved=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError,match="innym oknie"):
        core.set_text(s,"mt-one","GT:a0","AAAA")
    assert s.sidecar_path.read_bytes()==saved


def test_completed_snapshot_contains_canvas_and_is_invalidated_by_later_edit(tmp_path):
    from auto_annotation_tool.ranking.mobile_verification_snapshot import has_valid_ready_snapshot
    s=source_session(tmp_path);text_all(s);core.check_scene(s,"mt-one")
    s.complete()
    assert has_valid_ready_snapshot(s)
    assert s.review.review_status=="COMPLETED"
    frozen=list(s.sidecar_path.parent.glob("ready_for_evaluation/**/verified_review.json"))
    assert len(frozen)==1
    blob=frozen[0].read_bytes();d=json.loads(blob)
    assert d["invocation_annotations"]["mt-one"][core.KEY]["checked"]
    core.set_text(s,"mt-one","GT:a0","WA09999")
    assert not has_valid_ready_snapshot(s)
    assert frozen[0].read_bytes()==blob
    assert s.review.review_status=="IN_PROGRESS"


def test_missing_source_image_rejects_edits_without_changes(tmp_path):
    s=source_session(tmp_path);before=s.review.to_dict()
    s.entry_names.discard("samples/frames/source.jpg")
    with pytest.raises(ValueError):core.add_plate(s,"mt-one",MANUAL)
    assert s.review.to_dict()==before


def test_duplicate_detection_index_blocks_scene(tmp_path):
    s=source_session(tmp_path)
    s.attempts["a1"]["mt_detection_index"]=0
    with pytest.raises(ValueError):core.scene(s,"mt-one")
