"""Performance qualification, aggregation, and source protection in existing Z4."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from _private_test_data import private_test_path
from types import SimpleNamespace as NS
import json

import pytest

from auto_annotation_tool.gui.mobile_analysis import AnalysisInput, AnalysisCohort, SourceValidation, MetricEligibility
from auto_annotation_tool.gui import mobile_analysis_f5a as f5a


def source(rows=(), variant='R1', generation=1, left=53):
    descriptor=dict(source_width_px=960, source_height_px=1280, input_width_px=854, input_height_px=493,
        tv_bbox_px=dict(left=left, top=257, right=907, bottom=750), corners_norm=[{'x':.1,'y':.2}],
        processing_transform='CROP_ONLY_TRANSLATION', H_TV='NOT_APPLIED', digest_sha256=str(generation),
        calibration_digest_sha256=str(generation), source_geometry=dict(binding_generation=generation,
            viewport_revision=generation, rotation_degrees=0, zoom_ratio=1),
        calibration_snapshot=dict(digest_sha256=str(generation), frozen_at_utc=str(generation),
            source_geometry=dict(binding_generation=generation, viewport_revision=generation, rotation_degrees=0)))
    stages={k:dict(enabled=True,model_id=k,precision='fp32',cpu_threads=2) for k in ('mp','mt','mz')}
    stages['mp']['enabled']=variant!='R0'
    execution=dict(stages=stages, feature_flags=dict(vehicle_tracking=variant!='R0'),
        variant=variant, roi_budget_policy={'R0':'r0_full_frame','R1':'r1_one_roi','R2':'r2_two_roi'}[variant])
    coll=dict(session_id='session-'+variant, series_id='F5A-TEST', scenario_id='PAIRED_ROI_01', variant=variant,
        replicate_index=1, state='COMPLETED', collection_complete=True, input_mode='TV_BBOX_CROP',
        dropped_sample_count=0, dropped_telemetry_count=0, integrity_loss_count=0, execution=execution)
    raw=dict(experiment=dict(notes='BLOK_A · 10 zdjęć × 30 s'), session_duration_ms=300000,
        data_retention=dict(complete=True,trace_records_evicted=0), device=dict(model='phone'))
    bundle=NS(path=Path('synthetic.alprsession'), collection_session=coll, report_payload=raw,
        report=NS(raw=raw), manifest={}, scene_evidence={}, tv_bbox_evidence=dict(available=True,status='VALID',descriptor=descriptor))
    inp=AnalysisInput(None,coll['session_id'],'a'*64,'report','synthetic.alprsession',None,Path('unused'))
    result=SourceValidation(inp,'VERIFIED','',bundle=bundle,metadata=dict(experiment='F5A',variant=variant,
        identity_status='VERIFIED',apk_sha256='b'*64,trace_state='complete',series_id='F5A-TEST',
        scenario='PAIRED_ROI_01',is_pilot=False))
    f5a.project_timings(result,list(rows),'complete')
    return result


def row(frame=1, **values):
    return dict(frame_id=frame,status='recognized',plate_inference_ms=8,character_inference_ms=4,
        total_ms=20,mt_runs_this_frame=2,mz_runs=1,**values)


def test_technical_generations_are_not_material_and_original_is_untouched():
    a,b=source(generation=1),source(generation=99)
    before=deepcopy(b.bundle.tv_bbox_evidence)
    assert f5a.qualify([a,b])['status']=='PASS_DESCRIPTIVE'
    assert a.metadata['f5a_performance']['technical']!=b.metadata['f5a_performance']['technical']
    assert b.bundle.tv_bbox_evidence==before


@pytest.mark.parametrize('change,path',[
    (lambda s:s.bundle.tv_bbox_evidence['descriptor']['tv_bbox_px'].update(left=54),'descriptor.tv_bbox_px.left'),
    (lambda s:s.bundle.tv_bbox_evidence['descriptor']['source_geometry'].update(zoom_ratio=2),'descriptor.source_geometry.zoom_ratio'),
    (lambda s:s.bundle.collection_session['execution']['stages']['mt'].update(cpu_threads=4),'mt.cpu_threads'),
    (lambda s:s.metadata.update(apk_sha256='c'*64),'apk_sha256'),
])
def test_material_changes_are_explicit_and_never_silently_aggregated(change,path):
    a,b=source([row()],variant='R0'),source([row()],variant='R1')
    change(b);f5a.source_metadata(b)
    c=AnalysisCohort((),[a,b],'F5A',[],performance=f5a.qualify([a,b]))
    assert path in c.performance['material_differences']
    assert all(r['value'] is None for r in f5a.aggregate(c,'mt_trace_p50_ms','PAIRED_ROI_01'))
    assert b.metrics['mt_trace_p50_ms'].value==8  # retained local measurement


def test_multiple_roi_sum_does_not_become_per_call_time_or_sum_of_medians():
    s=source([row(),dict(row(frame=2),plate_inference_ms=12,total_ms=100)])
    assert s.metrics['mt_trace_p50_ms'].value==10
    assert s.metrics['pipeline_p50_ms'].value==60
    assert s.metrics['mt_p50_ms'].value is None
    assert 'MT_CALL_TIMING_UNPROVEN' in s.metrics['mt_p50_ms'].missing_reasons


def test_mp_only_cancelled_unknown_and_absent_counts_remain_distinct():
    rows=[dict(row(),mt_runs_this_frame=1),
        dict(frame_id=2,status='scan_queue_updated',vehicle_inference_ms=15,vehicle_runs=1,total_ms=30),
        dict(row(frame=3),status='scene_superseded',plate_inference_ms=999,mt_runs_this_frame=3),
        dict(row(frame=4),status='unknown_future_status',plate_inference_ms=999)]
    s=source(rows)
    assert s.metrics['pipeline_p50_ms'].value==20
    assert s.metrics['mp_trace_p50_ms'].value==15
    assert s.metrics['mt_p50_ms'].value==8
    assert s.metrics['mt_calls'].value==6  # only explicit counters, including cancelled/unknown status
    assert s.metrics['mt_calls'].n_scored==3 and s.metrics['mt_calls'].n_total_eligible==4
    assert s.metrics['mt_calls'].status=='SUBSET'
    assert s.metadata['f5a_performance']['trace_metrics']['mt_calls']['calls_in_cancelled_traces']==3
    assert s.metrics['mt_calls_per_exposure'].value is None


def test_disabled_mp_is_na_and_measured_zero_counter_is_zero():
    s=source([dict(row(),mz_runs=0,character_inference_ms='')],variant='R0')
    assert s.metrics['mp_trace_p50_ms'].value is None
    assert s.metrics['mp_trace_p50_ms'].missing_reasons==('MP_DISABLED',)
    assert s.metrics['mz_calls'].value==0
    assert s.metrics['mz_trace_p50_ms'].value is None


def test_incomplete_stream_cannot_supply_complete_time_or_counts():
    s=source([row()])
    f5a.project_timings(s,[row()],'incomplete')
    assert s.metrics['mt_trace_p50_ms'].value is None and s.metrics['mt_calls'].value is None


def test_equal_session_weight_does_not_weight_a_session_by_trace_count():
    a,b=source([dict(row(),plate_inference_ms=1)]*100),source([dict(row(),plate_inference_ms=9)])
    c=AnalysisCohort((),[a,b],'F5A',[],performance=f5a.qualify([a,b]))
    value=f5a.aggregate(c,'mt_trace_p50_ms','PAIRED_ROI_01')[1]
    assert value['value']==5 and value['n_sessions']==2 and value['n_source_observations']==101
    assert value['aggregation']=='median_of_session_p50'


def test_duplicate_or_pilot_cohort_fails_aggregate_qualification():
    s=source([row()])
    assert f5a.qualify([s],[{'reason':'duplicate'}])['status']=='NOT_COMPARABLE'
    other=deepcopy(s);other.metadata['is_pilot']=True
    assert 'F5A_SERIES_OR_PILOT_MISMATCH' in f5a.qualify([s,other])['missing_reasons']


def test_observed_runtime_change_is_material_even_when_requested_settings_match():
    a,b=source(),source()
    a.bundle.report_payload['runtime_execution_evidence']=dict(stages=dict(plate=dict(backend_effective='LiteRT/CPU')))
    b.bundle.report_payload['runtime_execution_evidence']=dict(stages=dict(plate=dict(backend_effective='LiteRT/GPU')))
    f5a.source_metadata(a);f5a.source_metadata(b)
    gate=f5a.qualify([a,b])
    assert 'runtime_observed.plate.backend_effective' in gate['material_differences']
    assert gate['status']=='NOT_COMPARABLE'


def test_figure_has_only_three_rows_and_no_overlapping_na_annotations():
    from auto_annotation_tool.gui.mobile_analysis_figure import make_figure
    from auto_annotation_tool.gui.mobile_analysis_export import figure_plan
    sources=[source([row()],variant=v) for v in ('R0','R1','R2') for _ in range(12)]
    c=AnalysisCohort((),sources,'F5A',[],performance=f5a.qualify(sources))
    fig=make_figure(c,'mt_trace_p50_ms','PAIRED_ROI_01')
    assert len(fig.axes[0].patches)==3 and len(fig.axes[0].texts)==3
    assert fig._alpr_analysis['gt_mode']=='NOT_APPLICABLE_TIMING'
    empty=make_figure(c,'mt_calls_per_exposure','PAIRED_ROI_01')
    assert not empty.axes[0].texts and not empty.axes[0].patches
    assert not list(figure_plan(c,['mt_calls_per_exposure']))


def test_f5a_labels_use_block_or_session_identity_and_no_f4_language():
    from auto_annotation_tool.gui.mobile_analysis_presentation import session_label,metric_labels,report_sections
    s=source([row()])
    assert session_label(s)=='R1 · blok A · r01'
    s.metadata['block']=None
    assert s.input.session_id in session_label(s)
    c=AnalysisCohort((),[s],'F5A',[],performance=f5a.qualify([s]))
    assert 'F4' not in '\n'.join(report_sections(c))
    assert 'ALPR_MT_TRACE' in metric_labels('pipeline_p50_ms','F5A')['title']


def test_real_two_generations_use_native_validation_and_do_not_read_gt(monkeypatch,tmp_path):
    from auto_annotation_tool.gui import mobile_analysis as analysis
    from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle
    from auto_annotation_tool.ranking.mobile_tv_bbox_evidence import cohort_conflict
    root=private_test_path('F5A-ROI-A12-01/02_sesje')
    paths=[root/'R0/PAIRED_ROI_01/r02/F5A_R0_r02_BLOK_A.alprsession',root/'R1/PAIRED_ROI_01/r01/F5A_R1_r01_BLOK_D.alprsession']
    if not all(p.is_file() for p in paths):pytest.skip('Local real F5A source fixtures unavailable')
    def no_gt(*args,**kwargs):raise AssertionError('Timing analysis must not evaluate GT')
    monkeypatch.setattr(analysis,'_quality',no_gt)
    inputs=[]
    for p in paths:
        bundle=read_mobile_report_bundle(p)
        assert bundle.validation.ok
        inputs.append(analysis.make_input(bundle.report,source_path=p,bundle=bundle,results_root=tmp_path,review_root=tmp_path))
    c=analysis.analyze(inputs,'F5A')
    assert c.performance['status']=='PASS_DESCRIPTIVE'
    assert cohort_conflict(s.metadata['input_context'] for s in c.sessions)=='TV_BBOX_FROZEN_INPUT_COHORT_MISMATCH'
    assert all(s.metrics['mt_trace_p50_ms'].value>0 for s in c.sessions)
    assert all('f4_variant' not in s.metadata and s.quality_state=='NOT_APPLICABLE_TIMING' for s in c.sessions)


def test_native_window_header_metric_selection_and_source_table():
    import tkinter as tk
    from auto_annotation_tool.gui.mobile_analysis_gui import UnifiedAnalysisWindow
    root=tk.Tk();root.withdraw()
    sources=[source([row()],variant=v) for v in ('R0','R1','R2') for _ in range(12)]
    c=AnalysisCohort((),sources,'F5A',[],performance=f5a.qualify(sources))
    window=None
    try:
        window=UnifiedAnalysisWindow(root,c)
        root.update()
        assert 'F5A' in window.protocol_label.cget('text') and 'F4' not in window.protocol_label.cget('text')
        assert len(window.sources.get_children())==36
        assert len(window.figure.axes[0].get_yticklabels())==3
        window.metric.current(window._metric_ids.index('mt_trace_p50_ms'));window._render();root.update()
        window._set_cohort(c);root.update()
        assert window.metric_id()=='mt_trace_p50_ms'
        assert 'REGISTRY_SHA_NOT_FOUND' not in window.details.get('1.0','end')
        assert 'F4' not in window.details.get('1.0','end')
        assert window.sources.cget('yscrollcommand')
    finally:
        if window is not None:window._close()
        root.destroy()
