"""ALPR-ANALYSIS-1.0: one intent basket, source and metric-specific trust."""
from pathlib import Path
from dataclasses import replace
import copy,csv,hashlib,json,shutil
import pytest

from _private_test_data import private_test_path
OUT=private_test_path('output/Z4_UNIFIED_ANALYSIS_F4_F6_20261031_002')
R02='exp-1792699303016-c8b05efb'
F32='exp-1792701720679-5318a7d2'

@pytest.fixture(scope='module')
def inputs():
    from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle
    from auto_annotation_tool.gui.mobile_analysis import make_input
    root=OUT/'private/series/03_wyniki';reviews=OUT/'private/catalog'
    inventory=OUT/'source_inventory.json'
    if not inventory.is_file():
        pytest.skip('Real F4 acceptance fixtures are local; see ALPR-ANALYSIS-1.0 handoff for the private source inventory.')
    rows=json.loads(inventory.read_text(encoding='utf-8'))
    return {r['session_id']:make_input(read_mobile_report_bundle(r['private_source']).report,
        source_path=r['private_source'],results_root=root,review_root=reviews) for r in rows}

def test_U13_r02_original_timing_has_own_denominators(inputs):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    result=analyze([inputs[R02]])
    s=result.sessions[0]
    assert s.quality_state=='APPROVED_LIMITED'
    assert s.metrics['mt_p50_ms'].value==pytest.approx(470.958231)
    assert s.metrics['mt_p95_ms'].n_scored==21
    assert s.metrics['mz_trace_p95_ms'].value==pytest.approx(1244.08064965)
    assert s.metrics['mz_trace_p95_ms'].n_scored==20
    assert s.metrics['pipeline_p95_ms'].value==pytest.approx(2775.115539)
    assert s.metrics['pipeline_p95_ms'].n_scored==21

def test_U24_real_subset_is_19_of_21_without_awarding_missing_units(inputs):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    s=analyze([inputs[R02]]).sessions[0]
    m=s.metrics['mt_f1_pct']
    assert m.status=='SUBSET' and m.n_scored==19 and m.n_total_eligible==21 and m.n_unscored==2
    assert m.value==pytest.approx(2*14/(2*14+8+5)*100)
    assert s.metrics['polygon_iou_pct'].status=='N/A'
    assert s.quality_state=='APPROVED_LIMITED'

def clone(inp,tmp_path):
    from auto_annotation_tool.gui.mobile_analysis import make_input
    source=tmp_path/'02_sesje'/Path(inp.source_path).name;source.parent.mkdir(parents=True);shutil.copyfile(inp.source_path,source)
    root=tmp_path/'03_wyniki';(root/'porownanie').mkdir(parents=True)
    folder=inp.results_root/inp.session_id
    if folder.is_dir():shutil.copytree(folder,root/inp.session_id)
    reviews=tmp_path/'catalog';(reviews/'human_reviews').mkdir(parents=True)
    side=inp.review_root/'human_reviews'/(inp.source_sha256+'.review.json')
    if side.exists():shutil.copyfile(side,reviews/'human_reviews'/side.name)
    return make_input(inp.report,source_path=source,results_root=root,review_root=reviews)

def test_U01_U02_selection_does_not_assert_full_quality(inputs):
    from auto_annotation_tool.gui.mobile_analysis import selection_identity,analyze
    for sid in (R02,F32):assert selection_identity(inputs[sid].report)==(sid,inputs[sid].source_sha256)
    c=analyze([inputs[F32],inputs[R02]])
    assert [s.quality_state for s in c.sessions]==['READY','APPROVED_LIMITED']
    assert c.sessions[0].metrics['exact_mz_pct'].status=='FULL'
    assert c.sessions[1].metrics['exact_mz_pct'].status=='N/A'

@pytest.mark.parametrize('sid,state',[('exp-1792702658896-47b4e60b','IN_PROGRESS'),('exp-1792702907090-d3edcf48','NONE')])
def test_U03_progress_and_unreviewed_sources_supply_time_not_quality(inputs,sid,state):
    from auto_annotation_tool.gui.mobile_analysis import analyze,selection_identity
    assert selection_identity(inputs[sid].report)
    s=analyze([inputs[sid]]).sessions[0]
    assert s.quality_state==state and s.metrics['mt_p95_ms'].value>0
    assert s.metrics['mt_f1_pct'].value is None and s.metrics['exact_mz_pct'].value is None

@pytest.mark.parametrize('fault',['missing','sha','zip','report_id'])
def test_U04_U07_U08_U09_invalid_sources_remain_diagnostic_without_numbers(inputs,tmp_path,fault):
    from auto_annotation_tool.gui.mobile_analysis import analyze,selection_identity
    inp=clone(inputs[R02],tmp_path);p=Path(inp.source_path)
    if fault=='missing':p.unlink()
    elif fault=='sha':p.write_bytes(p.read_bytes()+b'changed')
    elif fault=='zip':p.write_bytes(b'not a valid archive')
    else:inp=replace(inp,report=replace(inp.report,report_id='wrong-report-id'))
    assert selection_identity(inp.report)
    s=analyze([inp]).sessions[0]
    assert s.source_status in ('SOURCE_MISSING','SOURCE_MISMATCH','MANIFEST_INVALID')
    assert all(m.value is None and m.status=='N/A' for m in s.metrics.values())
    assert str(p) in s.source_reason

def test_U05_duplicates_are_not_independent_observations(inputs):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    c=analyze([inputs[F32],inputs[R02],inputs[R02]])
    assert len(c.sessions)==2 and len(c.duplicates)==1


def test_U09_internal_manifest_is_checked_even_when_outer_archive_SHA_matches(inputs,tmp_path):
    import zipfile
    from auto_annotation_tool.gui.mobile_analysis import analyze
    inp=clone(inputs[F32],tmp_path);p=Path(inp.source_path)
    with zipfile.ZipFile(p) as z:
        entries={name:z.read(name) for name in z.namelist()}
    manifest=json.loads(entries['manifest.json'])
    manifest['entry_sha256']['traces.csv']='0'*64
    entries['manifest.json']=json.dumps(manifest).encode('utf-8')
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in entries.items():z.writestr(name,data)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    inp=replace(inp,source_sha256=digest,report=replace(inp.report,source_archive_sha256=digest))
    s=analyze([inp]).sessions[0]
    assert s.source_status in ('SOURCE_MISMATCH','MANIFEST_INVALID')
    assert all(m.value is None for m in s.metrics.values())
    assert str(p) in s.source_reason

def test_U10_newer_invalid_export_does_not_reuse_older_quality(inputs,tmp_path):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    inp=clone(inputs[F32],tmp_path);folder=inp.results_root/inp.session_id
    export=next(p for p in folder.iterdir() if (p/'export_manifest.json').is_file())
    newer=folder/'review_r999999_export_001';shutil.copytree(export,newer)
    (newer/'canvas_review_summary.json').write_text('{}',encoding='utf-8')
    s=analyze([inp]).sessions[0]
    assert s.quality_state=='INVALID' and s.metrics['exact_mz_pct'].value is None
    assert s.source_status=='VERIFIED' and s.metrics['mt_p95_ms'].value>0

@pytest.mark.parametrize('sid,repeat,index',[('exp-1792702658896-47b4e60b','r01','4'),('exp-1792702907090-d3edcf48','r02','5'),('exp-1792703187662-931586a2','r03','6')])
def test_U11_verified_registry_maps_night_fp32_without_rewriting_source(inputs,sid,repeat,index):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    p=Path(inputs[sid].source_path);before=hashlib.sha256(p.read_bytes()).hexdigest()
    s=analyze([inputs[sid]]).sessions[0]
    assert s.metadata['repeat']==repeat and s.metadata['source_replicate_index']==index
    assert s.metadata['variant']=='DAY_FP32' and s.metadata['source_variant']=='NIGHT_F32'
    assert s.metadata['scenario']=='PAIR_NIGHT_01'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==before

def test_U12_frozen_matching_keeps_RAW_and_separate_invocations_distinct():
    from test_z4_mt_detection_evaluation import unit
    from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as mt
    a=unit();a['n_raw_suppressed']=12
    r=mt.score_unit(a)
    assert (r['tp'],r['fp'],r['fn'],r['n_raw_suppressed'])==(1,0,0,12)
    b=copy.deepcopy(a);b['unit_id']='second';b['mt_invocation_id']='other'
    combined=mt.aggregate_units([r,mt.score_unit(b)])
    assert combined['n_total']==2 and combined['tp']==2

def test_U14_U15_missing_MZ_is_NA_and_call_counts_differ_from_traces(inputs,monkeypatch):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    from auto_annotation_tool.gui import mobile_import_comparison as imp
    actual=analyze([inputs[R02]]).sessions[0]
    assert actual.metrics['mz_calls'].value==23
    assert actual.metrics['mz_trace_p95_ms'].n_scored==20
    assert actual.metrics['mz_call_p95_ms'].value is None
    original=imp._read_all_traces
    def no_mz(bundle):
        rows,status=original(bundle);return [dict(r,character_inference_ms=None,mz_runs='0') for r in rows],status
    monkeypatch.setattr(imp,'_read_all_traces',no_mz)
    s=analyze([inputs[R02]]).sessions[0]
    assert s.metrics['mz_trace_p95_ms'].value is None and s.metrics['mz_trace_p95_ms'].n_scored==0
    assert s.metrics['mt_p95_ms'].n_scored==21 and s.metrics['pipeline_p95_ms'].n_scored==21

def test_U16_U17_six_day_sessions_and_separate_night_panels(inputs):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    from auto_annotation_tool.gui.mobile_analysis_export import figure_plan
    c=analyze(list(inputs.values()));day=[s for s in c.sessions if s.metadata['scenario']=='PAIR_DAY_01']
    assert len(day)==6 and {s.metadata['repeat'] for s in day}=={'r01','r02','r03'}
    assert len({s.input.session_id for s in day})==6
    assert {scenario for scenario,_ in figure_plan(c)}=={'PAIR_DAY_01','PAIR_NIGHT_01'}
    assert c.interpretation=='DESCRIPTIVE'

def test_U18_incomplete_stream_does_not_use_preview_or_zero(inputs,monkeypatch):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    from auto_annotation_tool.gui import mobile_import_comparison as imp
    monkeypatch.setattr(imp,'_read_all_traces',lambda bundle:([],'INCOMPLETE_TRACE_SOURCE'))
    s=analyze([inputs[R02]]).sessions[0]
    for key in ('mt_p95_ms','mz_trace_p95_ms','pipeline_p95_ms'):
        m=s.metrics[key];assert m.value is None and 'INCOMPLETE_TRACE_SOURCE' in m.missing_reasons


@pytest.mark.parametrize('fault',['nonfinite_MZ','runtime_error'])
def test_U18_invalid_measurements_do_not_become_zero_or_complete_results(inputs,monkeypatch,fault):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    from auto_annotation_tool.gui import mobile_import_comparison as imp
    original=imp._read_all_traces
    def invalid(bundle):
        rows,status=original(bundle)
        change={'character_inference_ms':'nan'} if fault=='nonfinite_MZ' else {'status':'runtime_error'}
        return [dict(row,**change) for row in rows],status
    monkeypatch.setattr(imp,'_read_all_traces',invalid)
    s=analyze([inputs[R02]]).sessions[0]
    mz=s.metrics['mz_trace_p95_ms']
    assert mz.value is None and mz.status=='N/A' and mz.n_scored==0
    assert 'STAGE_NOT_RUN_OR_INVALID_TRACE' in mz.missing_reasons
    if fault=='nonfinite_MZ':
        assert s.metrics['mt_p95_ms'].n_scored==s.metrics['pipeline_p95_ms'].n_scored==21
    else:
        assert s.metrics['mt_p95_ms'].value is None and s.metrics['pipeline_p95_ms'].value is None

def test_U19_READY_quality_exactly_matches_existing_validated_counts(inputs):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    from auto_annotation_tool.gui.mobile_report_canvas_view import selected_canvas_export
    inp=inputs[F32];gate,review=selected_canvas_export(inp.session_id,inp.source_sha256,inp.results_root,inp.review_root)
    assert gate.ready
    s=analyze([inp]).sessions[0];c=review.canvas
    assert s.metrics['exact_mz_pct'].value==pytest.approx(c['exact_reads']/c['evaluated_reads']*100)
    assert s.metrics['micro_cer_pct'].value==pytest.approx(c['text_errors']/c['gt_characters']*100)
    assert s.metrics['polygon_iou_pct'].value==c['polygon_iou_mean_pct']
    assert s.metrics['exact_mz_pct'].gt_origin=='ASSISTED' and s.quality_state=='READY'
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_session
    canonical=evaluate_session(MobileReviewSession.open(inp.source_path,sidecar_path=review.directory/'review.json'))
    for key in ('tp','fp','fn'):assert s.metrics['mt_'+key].value==canonical[key]

def test_U20_U21_r02_missing_bbox_and_GTR_have_no_imputed_scores(inputs):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    s=analyze([inputs[R02]]).sessions[0];units={u['label']:u for u in s.quality_units}
    for label,reason in (('Obraz 07','FINAL_SOURCE_BBOX_UNAVAILABLE'),('Obraz 08','GT_GEOMETRY_UNCONFIRMED')):
        u=units[label];assert reason in u['missing_reasons']
        assert all(u[k] is None for k in ('tp','fp','fn'))
    assert s.metrics['polygon_iou_pct'].value is None and s.metrics['nmce_pct'].value is None

def test_U22_INVALID_QUAD_does_not_prevent_bbox_matching_or_invent_pose():
    from test_z4_mt_detection_evaluation import unit
    from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as mt
    u=unit();u['finals'][0]['pose_status']='DETECTION_INVALID_QUAD'
    r=mt.score_unit(u)
    assert r['tp']==1 and u['finals'][0]['pose_status']=='DETECTION_INVALID_QUAD'
    assert 'original_mt_quad' not in r['matched_pairs'][0]

def test_U23_MZ_NOT_RUN_has_no_conditional_read_error():
    from test_z4_akt87_verified_mt_mz import canvas_example
    from auto_annotation_tool.gui.mobile_compare_review_gate import paired_geometry_text
    c=canvas_example();row=c['rows'][0];row['mz_status']='NOT_RUN'
    assert list(paired_geometry_text(c))==[]

@pytest.mark.parametrize('change',['revision','same_revision_GTZ','missing_sidecar'])
def test_U25_quality_changes_invalidate_only_GT_dependent_values(inputs,tmp_path,change):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    inp=clone(inputs[F32],tmp_path);before=analyze([inp]).sessions[0]
    side=inp.review_root/'human_reviews'/(inp.source_sha256+'.review.json')
    if change=='missing_sidecar':side.unlink()
    else:
        d=json.loads(side.read_text(encoding='utf-8-sig'))
        if change=='revision':d['review_revision']+=1
        else:d['note']='Changed GT record without increment; deliberate invalid input'
        side.write_text(json.dumps(d),encoding='utf-8')
    now=analyze([inp]).sessions[0]
    assert now.metrics['mt_p95_ms'].value==before.metrics['mt_p95_ms'].value
    assert now.metrics['exact_mz_pct'].value is None and now.quality_state=='STALE'

def test_U26_U27_U28_U29_descriptive_limits_are_not_causal_or_scene_E2E(inputs):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    c=analyze([inputs[F32],inputs[R02]]);payload=c.payload();encoded=json.dumps(payload,ensure_ascii=False)
    assert c.interpretation=='DESCRIPTIVE' and 'NIE ZATWIERDZONO' in encoded
    assert 'termiczny' in encoded.lower() and 'APK' in encoded
    assert any('podzbiory' in x for x in payload['limitations'])
    for s in c.sessions:
        assert s.metrics['exact_e2e_pct'].value is None and 'SCENE_UNMAPPED' in s.metrics['exact_e2e_pct'].missing_reasons
        values=[]
        for row in s.bundle.trace_rows:
            try:
                value=float(row.get('pss_end_kb',''))
                if value>=0:values.append(value/1024)
            except (ValueError,TypeError):pass
        if values:
            assert s.metrics['pss_peak_mib'].value==max(values)
            assert s.metrics['pss_peak_mib'].n_scored==len(values)
        else:assert s.metrics['pss_peak_mib'].value is None

@pytest.mark.parametrize('question',['F5A','F5B'])
def test_U30_missing_F5_is_not_a_fabricated_variant(inputs,question):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    c=analyze([inputs[F32],inputs[R02]],question)
    assert all(m.value is None and m.status=='NOT_COMPARABLE' for s in c.sessions for m in s.metrics.values())
    assert all(m.n_scored is None for s in c.sessions for m in s.metrics.values())
    with pytest.raises(ValueError,match='F6'):analyze(c.inputs,'F6')

def test_U32_U33_plot_contains_both_sources_and_explicit_missing_values(inputs):
    from auto_annotation_tool.gui.mobile_analysis import analyze,PROTOCOL_ID
    from auto_annotation_tool.gui.mobile_analysis_export import make_figure
    c=analyze([inputs[F32],inputs[R02]])
    fig=make_figure(c,'exact_mz_pct','PAIR_DAY_01')
    assert len(fig.axes[0].get_yticklabels())==2 and len(fig.axes[0].patches)==1
    assert fig._alpr_analysis['values'][1] is None
    assert fig._alpr_analysis['sources']==[inputs[F32].source_sha256,inputs[R02].source_sha256]
    text=' '.join(x.get_text() for x in fig.texts+fig.axes[0].texts)
    assert PROTOCOL_ID in text and 'N/A' in text and 'braki=' in text and 'DESCRIPTIVE' in text

def test_U34_export_six_DAY_has_consistent_values_SHA_and_new_folder(inputs,tmp_path):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    from auto_annotation_tool.gui.mobile_analysis_export import export_analysis
    all_=analyze(list(inputs.values()));day=[s.input for s in all_.sessions if s.metadata['scenario']=='PAIR_DAY_01']
    cohort=analyze(day);root=tmp_path/'03_wyniki';(root/'porownanie').mkdir(parents=True)
    dest,fresh=export_analysis(cohort,root,metrics=('mt_p95_ms','mt_f1_pct','exact_mz_pct'))
    manifest=json.loads((dest/'analysis_manifest.json').read_text(encoding='utf-8'))
    assert manifest['analysis_protocol_id']=='ALPR-ANALYSIS-1.0' and len(manifest['sources'])==6
    assert len(list(dest.glob('*.png')))==3 and len(list(dest.glob('*.svg')))==3
    for item in manifest['files']:assert hashlib.sha256((dest/item['name']).read_bytes()).hexdigest()==item['sha256']
    csv_rows=list(csv.DictReader((dest/'readiness_matrix.csv').open(encoding='utf-8-sig'),delimiter=';'))
    rows={r['metric_id']:r for r in csv_rows if r['session_id']==R02}
    assert rows['exact_mz_pct']['value']=='' and rows['exact_mz_pct']['status']=='N/A'
    assert rows['mt_f1_pct']['n_scored']=='19' and rows['mt_f1_pct']['n_unscored']=='2'
    original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}
    second,_=export_analysis(cohort,root,metrics=('mt_p95_ms',))
    assert second!=dest and original=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}
    assert not list((root/'porownanie').glob('.alpr_analysis_stage_*'))

@pytest.mark.parametrize('when,change',[('before','source'),('during','source'),('during','review')])
def test_U35_TOCTOU_refuses_stale_numbers_and_removes_stage(inputs,tmp_path,monkeypatch,when,change):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    from auto_annotation_tool.gui import mobile_analysis_export as exp
    inp=clone(inputs[F32],tmp_path/'copy');cohort=analyze([inp,inputs[R02]])
    def mutate():
        if change=='source':
            p=Path(inp.source_path);p.write_bytes(p.read_bytes()+b'changed')
        else:
            p=inp.review_root/'human_reviews'/(inp.source_sha256+'.review.json');d=json.loads(p.read_text(encoding='utf-8-sig'));d['review_revision']+=1;p.write_text(json.dumps(d),encoding='utf-8')
    if when=='before':mutate()
    else:
        old=exp.make_figure
        def figure(*a,**kw):
            f=old(*a,**kw);mutate();return f
        monkeypatch.setattr(exp,'make_figure',figure)
    with pytest.raises(ValueError):exp.export_analysis(cohort,inp.results_root,metrics=('mt_p95_ms',))
    assert not list((inp.results_root/'porownanie').iterdir())

def test_U25_U35_export_after_GT_change_keeps_time_and_drops_quality(inputs,tmp_path):
    from auto_annotation_tool.gui.mobile_analysis import analyze
    from auto_annotation_tool.gui.mobile_analysis_export import export_analysis
    inp=clone(inputs[F32],tmp_path);cohort=analyze([inp,inputs[R02]])
    side=inp.review_root/'human_reviews'/(inp.source_sha256+'.review.json');d=json.loads(side.read_text(encoding='utf-8-sig'));d['review_revision']+=1
    side.write_text(json.dumps(d),encoding='utf-8')
    path,fresh=export_analysis(cohort,inp.results_root,metrics=('mt_p95_ms',))
    s=fresh.sessions[0]
    assert s.metrics['mt_p95_ms'].value>0 and s.metrics['exact_mz_pct'].value is None
    assert path.is_dir()
