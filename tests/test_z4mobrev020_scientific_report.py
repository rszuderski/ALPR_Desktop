"""Z4MOBREV020: scientific denominator/eligibility, stage statuses, export & UI contract."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_annotation_tool.ranking.mobile_scientific_review import (
    _rate_text, display_scientific_summary, make_scientific_summary,
    scientific_markdown, session_conditions_text, subject_stage, write_scientific_companion,
)
from auto_annotation_tool.gui import z4_mobile_sample_review as gui


MT_ENTRY = "samples/evidence/attempt-5.jpg"


def scenario(*, cardinality="one", geometry=True, mz_run=False, identified=False,
             no_detection=True, other_pending=False, assisted=True, mode="review"):
    status = "NO_DETECTION" if no_detection else "VALID_QUAD"
    aid = "attempt-5"
    group = SimpleNamespace(mt_invocation_id="mt-5", executed=True, cancelled=False,
                            execution_failed=False, records=[{"id":aid,"mt_status":status}],
                            evidence_entries=[MT_ENTRY])
    attempt = {"id":aid,"mt_status":status,"mt_invocation_id":"mt-5",
               "mt_input_evidence_entry": MT_ENTRY, "input_width":"512","input_height":"512",
               "mz_status":"READ" if mz_run else "NOT_RUN"}
    detail = {"pose_gt_schema":"manual_mt_input_quad_v1",
              "pose_gt_image_entry":MT_ENTRY,"pose_gt_input_size_px":[512.0,512.0],
              "pose_gt_quad_mt_input_px":[[100.,100.],[200.,100.],[205.,130.],[105.,130.]]}
    if not no_detection:detail.update({"evaluable":True,"is_plate":True})
    sess = SimpleNamespace(
        review=SimpleNamespace(subjects={"subject-5":{"ground_truth":"ABC123" if identified else "",
                                                          "evaluable":True if identified else None}},
                               attempt_annotations={aid:detail if geometry else {}},
                               source_archive_sha256="x"*64,review_mode="assisted" if assisted else "blinded_gt_v1",
                               review_status="IN_PROGRESS",review_revision=3),
        session_id="session-science",attempts={aid:attempt},mt_invocations={"mt-5":group},
        invocation_by_attempt={aid:"mt-5"},
        subjects={"subject-5":{"attempts":[aid],"samples":["sample-5"] if mz_run else []}},
        samples={"sample-5":{"attempt_id":aid,"mz_status":"READ"}} if mz_run else {},
        bundle=SimpleNamespace(experiment_session={"scenario_id":"TV_A","variant":"F4-TV-FP32-B",
                                                  "app_build":{"build_type":"debug"},
                                                  "execution":{"stages":{"mt":{"delegate":"cpu"},"mz":{"delegate":"cpu"}}}},
                               model_refs={"plate":{"model_display_id":"MT-C","precision":"fp32"},
                                           "character":{"model_display_id":"MZ-C","precision":"fp32"}},
                               report=SimpleNamespace(device={"name":"Galaxy A12","android_version":"12"})))
    sess.invocation_annotation=lambda _id: {"visible_plate_count":cardinality,"evaluable":True}
    sess.invocation_completion_issues=lambda _id: ([] if cardinality!="one" or no_detection
                                                     or bool(sess.review.attempt_annotations[aid].get("is_plate"))
                                                     else ["missing detection decision"])
    geoms=[{"attempt_id":aid,"metric_status":
                "missed_detection_gt_saved" if no_detection and geometry else
                "measured" if not no_detection and geometry else
                "missing_manual_gt" if not no_detection else "not_evaluable",
            "eligible":not no_detection and geometry,
            "polygon_iou_pct":89.5 if not no_detection and geometry else None,
            "nmce_pct":4.2 if not no_detection and geometry else None}]
    stat={"summary":{
        "evaluable_mt_invocations":1 if cardinality=="one" else 0,
        "mt_successful_invocations":0 if no_detection else 1,
        "mt_no_detection_invocations":1 if no_detection else 0,
        "mt_false_detections":0,"mt_multi_plate_invocations":0,
        "mt_execution_error_invocations":0,
        "evaluable_reads":1 if mz_run and identified else 0,
        "exact_reads":1 if mz_run and identified else 0,
        "gt_characters":6 if mz_run and identified else 0,
        "incorrect_characters":0,"missing_characters":0,"extra_characters":0,
        "evaluable_subjects":1 if mz_run and identified else 0,
        "subjects_with_exact_read":1 if mz_run and identified else 0,
        "median_time_to_first_exact_ms":137.5 if mz_run and identified else None,
        "p90_time_to_first_exact_ms":None},
        "mt_invocations":[{"mt_invocation_id":"mt-5","executed":True,"cancelled":False,
                            "execution_failed":False,"included":cardinality=="one"}],
        "subjects":[{"subject_key":"subject-5", "pending_decisions":0 if identified else 1}],
        "character_confusion":[]}
    if not no_detection and geometry:
        # measured Pose has its own GT, correctly assessed if detector confirmed
        pass
    return sess, stat, geoms


def test_missed_one_plate_mt_is_done_without_ocr_and_without_cer(tmp_path):
    sess,stat,rows=scenario()
    report=make_scientific_summary(sess,stat,rows)
    assert report["subjects"]["subject-5"]["label"]=="MT ocenione"
    assert report["subjects"]["subject-5"]["mt_complete"]
    assert report["subjects"]["subject-5"]["mz_status"]=="MZ nie uruchomiono"
    assert report["progress"]["subjects_mt_done"]==1
    assert report["mt"]["single_plate_n"]==1
    assert report["mt"]["localized_n"]==0
    assert report["mt"]["no_detection_n"]==1
    assert report["mt"]["failed_localization_n"]==1
    assert report["mt"]["success_rate"]==0.0
    assert report["pose"]["gt_missed_plate_n"]==1
    assert report["pose"]["measured_n"]==0
    assert report["pose"]["polygon_iou_mean_pct"] is None
    assert report["pose"]["nmce_mean_pct"] is None
    assert report["mz"]["cer"] is None
    assert report["mz"]["exact_rate"] is None
    assert report["alpr"]["exact_rate"] is None
    assert report["scientific_state"]=="MT_STAGE_COMPLETE_REVIEW_OPEN"
    assert "asystowana" in " ".join(report["caveats"]).lower()
    assert "nie jest to pełny recall" in " ".join(report["methods"])
    names=write_scientific_companion(report,tmp_path)
    assert len(names)==2
    doc=(tmp_path/'ocena_naukowa_session-science_rew0003.md').read_text(encoding='utf-8')
    assert '1/1' not in doc.split('## Wyniki z jawnymi mianownikami')[1].split('## Interpretacja')[0].split('| MT | Potwierdzone trafienia modelu |')[1].split('|')[0]
    assert '0/1 (0.00%)' in doc
    assert 'nieobliczalne' in doc
    assert 'nie jest to pełny recall' in doc
    assert json.loads((tmp_path/'ocena_naukowa_session-science_rew0003.json').read_text(encoding='utf-8'))['pose']['gt_missed_plate_n']==1


def test_missing_manual_gt_keeps_mt_pending():
    sess,stat,rows=scenario(geometry=False)
    r=make_scientific_summary(sess,stat,rows)
    assert not r['subjects']['subject-5']['mt_complete']
    assert r['subjects']['subject-5']['mt_pending']
    assert r['scientific_state']=='PRELIMINARY'


def test_detected_pose_real_metric_with_correct_denominator():
    sess,stat,rows=scenario(no_detection=False,geometry=True)
    r=make_scientific_summary(sess,stat,rows)
    assert r['mt']['success_rate']==1.0
    assert r['pose']['measured_n']==1
    assert r['pose']['polygon_iou_mean_pct']==pytest.approx(89.5)
    assert r['pose']['nmce_mean_pct']==pytest.approx(4.2)
    assert r['pose']['polygon_iou_median_pct']==pytest.approx(89.5)
    assert r['pose']['nmce_median_pct']==pytest.approx(4.2)
    assert r['subjects']['subject-5']['mt_complete']


def test_scientific_panel_does_not_fabricate_ocr_metrics():
    sess,stat,rows=scenario()
    body='\n'.join(txt for _,txt in display_scientific_summary(make_scientific_summary(sess,stat,rows)))
    assert 'Brak GT i ocenionego odczytu' in body or 'brak GT i ocenionego odczytu' in body
    assert 'CER: —' in body
    assert 'GT tablic pominiętych przez MT: 1' in body
    assert 'RAPORT WSTĘPNY' in body
    assert '0/1 (0.00%)' in body
    assert 'nie jest pełne recall/precision' in body


def test_multiple_plate_excluded_from_single_plate_success_rate():
    sess,stat,rows=scenario(cardinality='multiple')
    r=make_scientific_summary(sess,stat,rows)
    assert r['mt']['success_rate'] is None
    assert r['mt']['single_plate_n']==0
    assert 'brak mianownika' in _rate_text(0,0)


def test_ocr_measured_only_with_gt_and_actual_mz_run():
    sess,stat,rows=scenario(mz_run=True,identified=True)
    r=make_scientific_summary(sess,stat,rows)
    assert r['mz']['evaluable_reads']==1
    assert r['mz']['exact_rate']==1.0
    assert r['mz']['cer']==0.0
    assert r['alpr']['exact_rate']==1.0
    assert r['subjects']['subject-5']['mz_status']=='MZ ocenione'
    sess2,stat2,rows2=scenario(mz_run=True,identified=False)
    r2=make_scientific_summary(sess2,stat2,rows2)
    assert r2['subjects']['subject-5']['mz_pending']
    assert r2['mz']['cer'] is None


def test_rejected_gt_never_enters_pose_denominator():
    sess,stat,rows=scenario(no_detection=False,geometry=False)
    r=make_scientific_summary(sess,stat,rows)
    assert r['pose']['measured_n']==0
    assert r['pose']['polygon_iou_mean_pct'] is None
    assert r['pose']['nmce_mean_pct'] is None
    assert r['subjects']['subject-5']['mt_pending']


def test_science_export_keeps_unrelated_files_unchanged(tmp_path):
    sess,stat,rows=scenario()
    old=tmp_path/'review.json'; old.write_bytes(b'ORIGINAL_GT_PAYLOAD')
    saved=old.read_bytes()
    report=make_scientific_summary(sess,stat,rows)
    write_scientific_companion(report,tmp_path)
    assert old.read_bytes()==saved
    assert not (tmp_path/'ocena_naukowa_session-science_rew0003.md.tmp').exists()
    assert not (tmp_path/'ocena_naukowa_session-science_rew0003.json.tmp').exists()
    assert sess.review.attempt_annotations['attempt-5']['pose_gt_quad_mt_input_px'][0]==[100.,100.]


def test_provenance_panel_only_has_key_parameters_not_training_paths():
    sess,stat,rows=scenario()
    report=make_scientific_summary(sess,stat,rows)
    text=session_conditions_text(report['conditions'])
    assert 'Galaxy A12' in text and 'MT-C' in text and 'MZ-C' in text
    assert 'TV_A' in text and 'F4-TV-FP32-B' in text
    assert report['conditions']['models']['MT']['model_sha256']=='brak danych'
    assert report['source_archive_sha256']=="x"*64


def test_no_rewriting_raw_mt_or_mz_on_science_report():
    sess,stats,rows=scenario(no_detection=False)
    snapshot=deepcopy(sess.attempts)
    decisions=deepcopy(sess.review.attempt_annotations)
    for _ in range(2):make_scientific_summary(sess,stats,rows)
    assert sess.attempts==snapshot and sess.review.attempt_annotations==decisions


def test_gui_includes_scholarly_report_export_and_keeps_advanced_data():
    import inspect
    assert 'write_scientific_companion' in inspect.getsource(gui.MobileSampleReviewWindow.export)
    assert 'self.stats_tree.insert' in inspect.getsource(gui.MobileSampleReviewWindow._refresh_stats)
    assert 'Wyniki naukowe' in inspect.getsource(gui.MobileSampleReviewWindow._build)
    assert 'Warunki i źródła' in inspect.getsource(gui.MobileSampleReviewWindow._build)


def test_stage_does_not_manufacture_full_session_completion():
    sess,stats,rows=scenario()
    report=make_scientific_summary(sess,stats,rows)
    assert report['subjects']['subject-5']['mt_complete']
    assert sess.review.review_status=='IN_PROGRESS'
    assert report['scientific_state']!='FINAL_FULL_REVIEW'


def test_native_phone_latency_keeps_mt_mz_pipeline_separate():
    sess,stat,rows=scenario()
    sess.bundle.report_payload = {
        "latency": {
            "mt": {"count": 8, "median_ms":755.1293075, "p95_ms":869.4621809},
            "mz": {"count": 6, "median_ms":1020.3470385, "p95_ms":1387.330096},
            "pipeline": {"count": 8, "median_ms":2396.0502305, "p95_ms":3654.637280849999}},
        "frame_flow": {"frames_received":15, "frames_processed":8, "frames_skipped_frame_gate":7,
                       "estimated_upstream_gaps":2},
    }
    report=make_scientific_summary(sess,stat,rows)
    assert report['performance']['latency']['mt']['n']==8
    assert report['performance']['latency']['mz']['p50_ms']==pytest.approx(1020.3470385)
    assert report['performance']['latency']['pipeline']['p95_ms']==pytest.approx(3654.637280849999)
    assert report['performance']['frame_flow']['processed']==8
    assert report['performance']['frame_flow']['skipped_frame_gate']==7
    assert report['mz']['evaluable_reads']==0 # latency measured independently of GT
    assert 'przetworzono 8' in ' '.join(t for _,t in display_scientific_summary(report))
    assert 'P95' in scientific_markdown(report)


def test_latency_absent_or_zero_count_never_fabricates_zero_latency():
    sess,stat,rows=scenario()
    sess.bundle.report_payload = {'latency':{'mt':{'count':0,'median_ms':0.0,'p95_ms':0.0}}}
    report=make_scientific_summary(sess,stat,rows)
    assert report['performance']['latency']['mt']['p50_ms'] is None
    assert report['performance']['latency']['mz']['p50_ms'] is None
    assert report['performance']['frame_flow']['received'] is None


def test_dirty_apk_provenance_warning_shown():
    sess,stat,rows=scenario()
    sess.bundle.experiment_session['app_build']['git_dirty']=True
    report=make_scientific_summary(sess,stat,rows)
    assert any('git_dirty=true' in issue for issue in report['caveats'])
