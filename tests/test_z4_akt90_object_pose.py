"""P0: object detection, Pose availability and negative ROI are separate axes."""
import copy
import csv
from dataclasses import replace
import json

import pytest

from test_mobile_p2b2_canvas_review import source_session, text_all, MANUAL
from test_z4_akt89lm_resolution import invalid_source
from test_mobile_human_review import load
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_resolution import (
    link_manual_to_mt, keep_separate, confirm_no_plates, exclude_missing_source_pose,
)
from auto_annotation_tool.ranking.mobile_canvas_presentation import execution_state, focused_assessment
from auto_annotation_tool.gui.mobile_compare_review_gate import inspect_canvas, paired_geometry_text

GID='mt-one'
def accept(s):core.check_scene(s,GID);return core.summary(s)
def verdict(s):return inspect_canvas(core.summary(s),'COMPLETED')


def test_fn_without_mz_has_no_cer_or_geometry(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);core.add_plate(s,GID,MANUAL)
    d=accept(s)
    assert (d['missed_plates'],d['linked_plates'],d['geometry_n'],d['evaluated_reads'])==(1,0,0,0)
    assert d['rows'][0]['cer'] is d['rows'][0]['nmce_pct'] is d['rows'][0]['polygon_iou_pct'] is None
    assert verdict(s)[0]


def test_empty_inventory_is_not_automatic_tn_but_explicit_negative_is(tmp_path):
    s=source_session(tmp_path,n=0,mz=False)
    assert core.group_issues(s,GID) and not verdict(s)[0]
    core.set_scene_review_state(s,GID,checked=True)
    d=core.summary(s)  # explicit inspection is the negative decision; no second checkbox
    assert d['true_negative_frames']==d['negative_frames']==1
    assert d['expected_plates']==d['false_positive_mt']==d['missed_plates']==0
    assert not d['excluded_scenes'] and d['checked_scenes']==1 and d['geometry_n']==d['evaluated_reads']==0
    assert verdict(s)[0] and s.review.review_status=='IN_PROGRESS'
    reopened=load(s.path,tmp_path)
    assert core.summary(reopened)==d


def test_negative_with_detections_has_fps_no_gt_and_no_geometry(tmp_path):
    s=source_session(tmp_path,n=2);before=core.scene(s,GID)
    confirm_no_plates(s,GID);d=core.summary(s)
    assert (d['negative_frames_with_fp'],d['false_positive_mt'],d['true_negative_frames'])==(1,2,0)
    assert d['expected_plates']==d['geometry_n']==d['evaluated_reads']==0
    assert all(o['error'] and o['gt_text']=='' for o in core.scene(s,GID)['objects'])
    assert core.scene(s,GID)['parents']==before['parents'] and verdict(s)[0]


@pytest.mark.parametrize('missing_evidence',[False,True])
def test_same_object_without_pose_is_linked_with_separate_pose_reason(tmp_path,missing_evidence):
    s=invalid_source(tmp_path,missing_evidence=missing_evidence)
    manual=core.add_plate(s,GID,MANUAL)
    core.set_scene_review_state(s,GID,checked=True)
    before=core.scene(s,GID);link_manual_to_mt(s,GID,manual,'GT:a0')
    after=core.scene(s,GID);d=core.summary(s)
    assert after['checked'] and not core.group_issues(s,GID)  # one contextual decision fills the gap
    assert len(after['objects'])==1 and after['parents']==before['parents'] and after['parents'][0]['quad'] is None
    assert (d['expected_plates'],d['linked_plates'],d['missed_plates'],d['false_positive_mt'])==(1,1,0,0)
    row=d['rows'][0]
    assert row['pose_status']==('EVIDENCE_UNAVAILABLE' if missing_evidence else 'INVALID')
    assert row['geometry_availability']=='NOT_AVAILABLE'
    assert row['cer'] is row['nmce_pct'] is row['polygon_iou_pct'] is None
    assert d['geometry_n']==d['evaluated_reads']==d['gt_characters']==0
    assert verdict(s)[0] and not list(paired_geometry_text(d))
    assert s.review.review_status=='IN_PROGRESS'
    m=focused_assessment(after,'GT:a0',execution_state(s,GID))
    assert m['geometry_status']=='NOT_AVAILABLE' and 'trafne wskazanie' in m['frame']
    assert after['events'][-1]['object_before']['manual']==before['objects'][1]
    assert core.summary(load(s.path,tmp_path))==d
    paths=core.export(s,tmp_path/'export')
    assert json.loads(paths[0].read_text())==d
    assert 'NOT_AVAILABLE' in paths[1].read_text(encoding='utf-8-sig')
    assert 'Tożsamość wskazania MT i jakość Pose są odrębne' in paths[2].read_text(encoding='utf-8')


def test_same_object_valid_pose_has_canonical_metrics_and_preserves_text(tmp_path):
    s=source_session(tmp_path,n=1);text_all(s)
    manual=core.add_plate(s,GID,MANUAL);original=copy.deepcopy(core.scene(s,GID)['parents'])
    link_manual_to_mt(s,GID,manual,'GT:a0');d=accept(s)
    expected=core.compare_mt_pose_to_gt(original[0]['quad'],MANUAL,1000,700)
    assert d['rows'][0]['nmce_pct']==expected['nmce_pct'] and d['geometry_n']==1
    assert core.scene(s,GID)['parents']==original and core.scene(s,GID)['objects'][0]['gt_text']=='WA01234'
    assert verdict(s)[0]


def test_different_is_not_auto_fp_or_fn_before_full_review(tmp_path):
    s=invalid_source(tmp_path);manual=core.add_plate(s,GID,MANUAL)
    keep_separate(s,GID,manual,'GT:a0')
    d=core.summary(s)
    assert len(core.scene(s,GID)['objects'])==2
    assert not core.scene(s,GID)['objects'][0]['error']
    assert d['false_positive_mt']==d['missed_plates']==0 and core.group_issues(s,GID)
    core.toggle_error(s,GID,'GT:a0');d=accept(s)
    assert (d['false_positive_mt'],d['missed_plates'],d['linked_plates'])==(1,1,0) and verdict(s)[0]


def test_two_real_plates_one_mt_remain_distinct(tmp_path):
    s=source_session(tmp_path,n=1);text_all(s);manual=core.add_plate(s,GID,MANUAL)
    keep_separate(s,GID,manual,'GT:a0');d=accept(s)
    assert (d['linked_plates'],d['missed_plates'],d['geometry_n'],d['evaluated_reads'])==(1,1,1,1)
    assert verdict(s)[0]


def test_nms_raw_cannot_be_matched_or_counted_as_fp(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking import mobile_mt_postdedup
    from test_z4_p0_physical_gt import session_fixture
    s=session_fixture(tmp_path,n=2,suppressed=1)
    core.set_text(s,GID,'GT:a0','WA01234');manual=core.add_plate(s,GID,MANUAL)
    before=s.review.to_dict()
    with pytest.raises(ValueError):link_manual_to_mt(s,GID,manual,'GT:a1')
    assert s.review.to_dict()==before
    link_manual_to_mt(s,GID,manual,'GT:a0');d=accept(s)
    assert (d['linked_plates'],d['raw_suppressed_mt'],d['false_positive_mt'])==(1,1,0)


@pytest.mark.parametrize('case',['not_run','cancelled','missing_frame'])
def test_missing_execution_or_source_cannot_make_tn_or_link(tmp_path,case):
    s=source_session(tmp_path,n=0,mz=False);g=s.mt_invocations[GID]
    if case=='not_run':
        for r in g.records:r.update(mt_status='NOT_RUN',mt_executed=False)
        s.mt_invocations[GID]=replace(g,executed=False)
    elif case=='cancelled':s.mt_invocations[GID]=replace(g,cancelled=True)
    else:
        for r in g.records:r['source_frame_status']='MISSING'
    before=s.review.to_dict()
    with pytest.raises(ValueError):confirm_no_plates(s,GID)
    assert s.review.to_dict()==before and core.summary(s)['missed_plates']==0


def test_reference_outside_roi_is_not_fn_of_this_invocation(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);before=s.review.to_dict()
    with pytest.raises(ValueError):core.add_plate(s,GID,[(5,5),(60,5),(60,25),(5,25)])
    assert s.review.to_dict()==before and core.summary(s)['missed_plates']==0


def test_absence_cannot_erase_manual_gt_and_pose_failure_cannot_exclude_image(tmp_path):
    s=invalid_source(tmp_path);core.add_plate(s,GID,MANUAL);before=s.review.to_dict()
    with pytest.raises(ValueError):confirm_no_plates(s,GID)
    with pytest.raises(ValueError,match='Sam brak Pose'):exclude_missing_source_pose(s,GID)
    assert s.review.to_dict()==before


@pytest.mark.parametrize('corruption',['missing_decision','wrong_sha','wrong_attempt','fake_iou','wrong_count','unknown_pose'])
def test_gate_rejects_unproven_unavailable_geometry(tmp_path,corruption):
    s=invalid_source(tmp_path);manual=core.add_plate(s,GID,MANUAL)
    link_manual_to_mt(s,GID,manual,'GT:a0');d=accept(s);assert verdict(s)[0]
    d=copy.deepcopy(d);row=d['rows'][0]
    if corruption=='missing_decision':row.pop('identity_decision')
    elif corruption=='wrong_sha':row['identity_decision']['source_archive_sha256']='a'*64
    elif corruption=='wrong_attempt':row['identity_decision']['attempt_id']='other'
    elif corruption=='fake_iou':row['polygon_iou_pct']=0
    elif corruption=='wrong_count':d['pose_invalid_matches']=0
    else:row['pose_status']='VALID'
    assert not inspect_canvas(d,'COMPLETED')[0]


def test_tn_exports_frame_unit_and_no_fake_plate(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);confirm_no_plates(s,GID)
    files=core.export(s,tmp_path/'export')
    assert len(files)==5 and files[3].name=='canvas_review_frames.csv'
    assert files[4].name=='canvas_physical_plates.csv'
    with files[4].open(encoding='utf-8-sig',newline='') as f:assert not list(csv.DictReader(f))
    with files[3].open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['frame_outcome']=='TN' and rows[0]['expected']=='0'
    assert rows[0]['source_archive_sha256']==s.bundle.source_archive_sha256
    assert rows[0]['review_revision']==str(s.review.review_revision)
    with files[1].open(encoding='utf-8-sig',newline='') as f:assert not list(csv.DictReader(f))
    assert 'TN na jednostce klatka/ROI | 1' in files[2].read_text(encoding='utf-8')
    bad=core.summary(s);bad['scenes']*=2
    bad.update(scene_count=2,checked_scenes=2,negative_frames=2,true_negative_frames=2)
    assert not inspect_canvas(bad,'COMPLETED')[0]


def test_nmce_above_100_is_not_clamped_by_summary_or_gate(tmp_path):
    s=source_session(tmp_path,n=1);text_all(s)
    reference=[(700,450),(720,450),(720,460),(700,460)]
    core.correct_plate(s,GID,'GT:a0',reference);d=accept(s)
    assert d['rows'][0]['nmce_pct']>100 and d['nmce_mean_pct']>100
    assert d['rows'][0]['polygon_iou_pct']==0 and verdict(s)[0]
    assert list(paired_geometry_text(d))[0]['nmce_pct']>100


@pytest.mark.parametrize('damage',['unknown_count','not_executed','wrong_invocation','conflicting_trace'])
def test_source_identity_is_not_inferred_from_an_invalid_pose(tmp_path,monkeypatch,damage):
    from auto_annotation_tool.ranking import mobile_mt_postdedup as dedup
    s=invalid_source(tmp_path);manual=core.add_plate(s,GID,MANUAL)
    if damage=='unknown_count':s.mt_invocations[GID]=replace(s.mt_invocations[GID],detection_count=None)
    elif damage=='not_executed':s.attempts['a0']['mt_executed']=False
    elif damage=='wrong_invocation':s.attempts['a0']['mt_invocation_id']='other'
    else:
        row=s.attempts['a0']
        key=tuple(dedup._source_integer(row.get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation'))
        monkeypatch.setattr(dedup,'_trace_index',lambda *a:{key:[{'counters':{'plate_detections_after_dedup':0}}]})
    frozen=copy.deepcopy(s.review.to_dict())
    with pytest.raises(ValueError):link_manual_to_mt(s,GID,manual,'GT:a0')
    assert s.review.to_dict()==frozen


@pytest.mark.parametrize('negative',[False,True])
def test_v2_export_uses_existing_sha_revision_and_plot_gates(tmp_path,negative):
    from auto_annotation_tool.gui.mobile_compare_review_gate import imported_verification
    from auto_annotation_tool.gui.mobile_comparison_core import load_session_export,figures
    from test_z4_akt87_verified_mt_mz import make_export
    import matplotlib.pyplot as plt
    if negative:
        s=source_session(tmp_path,n=0,mz=False);confirm_no_plates(s,GID)
    else:
        s=invalid_source(tmp_path);manual=core.add_plate(s,GID,MANUAL)
        link_manual_to_mt(s,GID,manual,'GT:a0');accept(s)
    d=core.summary(s);sha=s.bundle.source_archive_sha256;rev=s.review.review_revision
    directory=make_export(tmp_path/'results',canvas=d,rev=rev,sid=s.session_id,sha=sha)
    review_root=tmp_path/'db';side=review_root/'human_reviews'/(sha+'.review.json');side.parent.mkdir(parents=True)
    payload=dict(session_id=s.session_id,source_archive_sha256=sha,review_revision=rev,review_status='COMPLETED')
    side.write_text(json.dumps(payload))
    record=load_session_export(directory)
    assert record.canvas_ready and record.canvas_cer_pct is None
    assert imported_verification(s.session_id,sha,tmp_path/'results',review_root).ready
    assert list(paired_geometry_text(record.canvas))==[]
    charts=list(figures([record]))
    try:
        assert charts
        assert record.canvas['geometry_n']==record.canvas['evaluated_reads']==0
    finally:
        for _,fig in charts:plt.close(fig)
    payload['review_revision']+=1;side.write_text(json.dumps(payload))
    assert imported_verification(s.session_id,sha,tmp_path/'results',review_root).state=='STALE'
    (directory/'canvas_review_summary.json').write_text('{}')
    assert imported_verification(s.session_id,sha,tmp_path/'results',review_root).state=='INVALID'
